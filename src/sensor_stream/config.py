from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class ReceiverConfig:
    transport: Literal["wifi", "usb"] = "wifi"
    usb_port: int = 2345
    usb_udid: str | None = None
    bind_host: str = "0.0.0.0"
    bind_port: int = 5678
    # Delivery limits protect the consumer queue; framing limits below protect
    # messages still being assembled from the transport.
    queue_capacity: int = 4096
    max_buffer_bytes: int = 64 * 1024 * 1024
    overflow_policy: Literal["raise", "drop_oldest"] = "raise"
    max_chunk_bytes: int = 1024 * 1024
    max_message_bytes: int = 16 * 1024 * 1024
    max_inflight_messages: int = 8
    max_chunks_per_message: int = 4096
    assembly_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.transport not in ("wifi", "usb") or not 1 <= self.usb_port <= 65535:
            raise ValueError("Use transport wifi/usb and USB port between 1 and 65535")
        if not self.bind_host or not 0 <= self.bind_port <= 65535:
            raise ValueError("Provide a host and port between 0 and 65535")
        for name in (
            "queue_capacity",
            "max_buffer_bytes",
            "max_chunk_bytes",
            "max_message_bytes",
            "max_inflight_messages",
            "max_chunks_per_message",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.assembly_timeout_s) or self.assembly_timeout_s <= 0:
            raise ValueError("assembly_timeout_s must be finite and positive")
        if self.overflow_policy not in ("raise", "drop_oldest"):
            raise ValueError("Unknown overflow policy")
