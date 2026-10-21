"""Add the diagnostic driver and compiler compatibility to an external ORB checkout."""

from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
args = parser.parse_args()
root = args.checkout.resolve()
header = root / "include/System.h"
text = header.read_text()
marker = "    int GetTrackingState();"
accessor = """
    // Read-only diagnostic access; tracking/map policies are unchanged.
    long unsigned int GetCurrentMapIdForDiagnostics() {
        return mpAtlas->GetCurrentMap()->GetId();
    }
"""
if "GetCurrentMapIdForDiagnostics" not in text:
    if marker not in text:
        raise ValueError("Unexpected upstream System.h")
    header.write_text(text.replace(marker, marker + "\n" + accessor))
cmake = root / "CMakeLists.txt"
text = cmake.read_text().replace("-std=c++11", "-std=c++14")
# New Pangolin headers require C++14. No algorithm modifications.
if "fusionlab_walking" not in text:
    text += """
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(fusionlab_walking fusionlab_walking.cc)
target_link_libraries(fusionlab_walking ${PROJECT_NAME})
"""
cmake.write_text(text)
shutil.copy2(Path(__file__).with_name("walking.cc"), root / "fusionlab_walking.cc")
print("Prepared read-only diagnostics driver; C++14 enabled for current Pangolin.")
