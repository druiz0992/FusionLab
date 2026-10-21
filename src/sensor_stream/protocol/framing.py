"""Incremental parser for SensorStream's 13-byte Wi-Fi chunk headers."""

from dataclasses import dataclass, field
from struct import Struct

from ..config import ReceiverConfig
from ..errors import ProtocolError

# Network byte order: message ID, chunk index, final-chunk flag, payload size.
HEADER = Struct("!IIBI")


@dataclass
class _Assembly:
    started: float
    chunks: dict[int, bytes] = field(default_factory=dict)
    size: int = 0
    last: int | None = None


class ChunkParser:
    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.buffer = bytearray()
        self.pending: dict[int, _Assembly] = {}
        self.partial_started: float | None = None

    def check_timeout(self, now: float) -> None:
        for message_id, assembly in self.pending.items():
            if now - assembly.started >= self.config.assembly_timeout_s:
                raise ProtocolError(
                    f"Message {message_id} assembly timed out: {len(assembly.chunks)} chunks, {assembly.size} bytes"
                )
        if (
            self.partial_started is not None
            and now - self.partial_started >= self.config.assembly_timeout_s
        ):
            raise ProtocolError(
                f"Partial TCP chunk timed out: {len(self.buffer)} buffered bytes"
            )

    def finish(self) -> None:
        if self.buffer or self.pending:
            raise ProtocolError(
                f"Connection ended mid-message: {len(self.buffer)} buffered bytes, {len(self.pending)} incomplete messages"
            )

    def feed(self, data: bytes, now: float) -> list[bytes]:
        self.check_timeout(now)
        # TCP reads can split a header/payload or contain several chunks.
        # Keep incomplete bytes until a later feed supplies the rest.
        self.buffer.extend(data)
        result: list[bytes] = []
        consumed = False
        while len(self.buffer) >= HEADER.size:
            message_id, index, last, size = HEADER.unpack_from(self.buffer)
            if size > self.config.max_chunk_bytes or last not in (0, 1):
                raise ProtocolError(
                    f"Invalid chunk header: message={message_id}, index={index}, last={last}, size={size}, limit={self.config.max_chunk_bytes}"
                )
            if index >= self.config.max_chunks_per_message:
                raise ProtocolError("Chunk index exceeds configured limit")
            if len(self.buffer) < HEADER.size + size:
                break
            payload = bytes(self.buffer[HEADER.size : HEADER.size + size])
            del self.buffer[: HEADER.size + size]
            consumed = True
            assembly = self.pending.get(message_id)
            if assembly is None:
                if len(self.pending) >= self.config.max_inflight_messages:
                    raise ProtocolError("Too many incomplete messages")
                assembly = self.pending[message_id] = _Assembly(now)
            if index in assembly.chunks:
                if assembly.chunks[index] != payload or bool(last) != (
                    assembly.last == index
                ):
                    raise ProtocolError("Conflicting duplicate chunk")
                continue
            if assembly.last is not None and index > assembly.last:
                raise ProtocolError("Chunk after final index")
            if last:
                if assembly.last is not None or any(i > index for i in assembly.chunks):
                    raise ProtocolError("Conflicting final chunk")
                assembly.last = index
            assembly.size += size
            if assembly.size > self.config.max_message_bytes:
                raise ProtocolError("Message exceeds configured limit")
            assembly.chunks[index] = payload
            # Indexes are nonnegative and bounded by the final index. Once all are
            # present, concatenate in index order even if chunks arrived out of order.
            if assembly.last is not None and len(assembly.chunks) == assembly.last + 1:
                result.append(
                    b"".join(assembly.chunks[i] for i in range(assembly.last + 1))
                )
                del self.pending[message_id]
        if self.buffer:
            # A completed old chunk can leave a different partial chunk behind.
            # Its deadline must not inherit the old chunk's start time during
            # continuous video where the buffer may never become empty.
            if consumed or self.partial_started is None:
                self.partial_started = now
        else:
            self.partial_started = None
        return result
