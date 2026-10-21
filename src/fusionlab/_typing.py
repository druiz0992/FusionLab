"""Type aliases used only in annotations; importing this never imports NumPy.

Modules write `from __future__ import annotations` and import these under
`if TYPE_CHECKING:`, so OpenCV/NumPy stay optional at runtime. Shapes, units and
frames are documented where each value is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    # A NumPy array of any shape and dtype.
    Array = NDArray[Any]
    # Normalized, undistorted camera rays keyed by persistent track ID.
    Rays = dict[int, NDArray[Any]]
