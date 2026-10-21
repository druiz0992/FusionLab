from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
# Frozen snapshots can be shared with reporting/preview code. The source
# replaces snapshots as counters change instead of mutating a shared object.
class SourceStats:
    connections: int = 0
    connected: bool = False
    bytes_received: int = 0
    messages_received: int = 0
    decoded_samples: int = 0
    decoded_imu_samples: int = 0
    decoded_gps_samples: int = 0
    decoded_camera_samples: int = 0
    decoded_depth_samples: int = 0
    observed_depth_rate_hz: float = 0.0
    observed_camera_rate_hz: float = 0.0
    observed_gps_rate_hz: float = 0.0
    malformed_messages: int = 0
    unsupported_messages: int = 0
    protocol_errors: int = 0
    transport_errors: int = 0
    last_error: str | None = None
    camera_relative_delay_s: float = 0.0  # Growth vs first frame, not absolute latency.
    rejected_connections: int = 0
    dropped_samples: int = 0
    timestamp_regressions: int = 0
    queue_depth: int = 0
    queue_high_water: int = 0
    last_receive_monotonic_ns: int | None = None
    observed_rate_hz: float = 0.0  # Session IMU rate by local receive time.
