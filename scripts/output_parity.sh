#!/bin/bash
# Behavior-neutral refactor check: run the pipeline commands with two source
# trees on the same real inputs and compare every output file byte for byte.
#
# Usage: scripts/output_parity.sh REFERENCE_SRC CANDIDATE_SRC WORK_DIR
#   REFERENCE_SRC  e.g. a `git worktree add --detach /tmp/ref HEAD` checkout's src/
#   CANDIDATE_SRC  usually ./src (the working tree)
#   WORK_DIR       new scratch directory for outputs (not under data/)
#
# Paths that name the output directory are normalized before comparing.
# Known nondeterminism (not covered here): checkerboard.pdf CreationDate and
# ~1e-10 OpenCV calibration noise.
set -euo pipefail
reference=$(realpath "$1"); candidate=$(realpath "$2"); work=$3
root=$(cd "$(dirname "$0")/.." && pwd)
walk=$root/data/sessions/walking/20261017T183215.982977Z-45b7f9c2
tracks=$walk/analysis/current/tracks/tracks.jsonl
calibration=$root/data/sessions/rgb_calibration/20261018T071611.049465Z-3cb5f83c/calibration/calibration.json

run_suite() {
    local src=$1 out=$2
    local py="env PYTHONPATH=$src MPLBACKEND=Agg $root/.venv/bin/python -m fusionlab"
    mkdir -p "$out"
    $py track-features "$walk" --output-dir "$out/tracks" --max-frames 40 > "$out/tracks.stdout"
    $py two-view "$walk" --tracks "$tracks" --frame-a 1000 --frame-b 1010 --output-dir "$out/two_view" > "$out/two_view.stdout"
    $py camera-motion --tracks "$tracks" --calibration "$calibration" --frame-a 1000 --frame-b 1010 --output-dir "$out/motion" > "$out/motion.stdout"
    vo() { local name=$1; shift; $py visual-odometry --tracks "$tracks" --calibration "$calibration" --output-dir "$out/$name" "$@" > "$out/$name.stdout"; }
    vo baseline --max-frames 120
    vo local_ba --max-frames 120 --landmark-policy refinement --bundle-adjustment
    vo refined --max-frames 150 --pose-acceptance refined
    vo parallax --max-frames 150 --keyframe-policy parallax
    vo validation --max-frames 150 --landmark-policy validation
    vo multiview --max-frames 150 --landmark-policy multiview --bundle-adjustment
    vo stride1 --max-frames 150 --stride 1 --landmark-policy refinement --bundle-adjustment
    $py diagnose-vo "$out/local_ba" --capture "$walk" --output-dir "$out/diag" > "$out/diag.stdout"
}

[ -e "$work" ] && { echo "WORK_DIR must be new: $work" >&2; exit 2; }
run_suite "$reference" "$work/reference"
run_suite "$candidate" "$work/candidate"
compared=0; differing=0
while IFS= read -r file; do
    compared=$((compared + 1))
    if ! cmp -s "$work/reference/$file" "$work/candidate/$file" \
        && ! diff -q <(sed "s|$work/reference|OUT|g" "$work/reference/$file") \
                     <(sed "s|$work/candidate|OUT|g" "$work/candidate/$file") > /dev/null; then
        echo "DIFFERS: $file"; differing=$((differing + 1))
    fi
done < <(cd "$work/reference" && find . -type f | sort)
echo "$compared files compared, $differing differ"
[ "$differing" -eq 0 ]
