"""Exceptions exposed by the receiver."""


class SensorStreamError(Exception):
    """Base receiver error."""


class TransportError(SensorStreamError):
    """Cannot open or operate the listener."""


class ProtocolError(SensorStreamError):
    """Invalid or excessive framing data."""


class BufferOverflowError(SensorStreamError):
    """Consumer did not keep up with the configured buffer."""
