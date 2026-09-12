# =============================================================================
# Data processing/prepare_data.py — Build the IES×APDA / IES×CINE feature
# table for one branch of the pipeline
# =============================================================================
# APP-UFT-only consolidation of Proyecto APDAs' three near-identical wrappers
# (Vector/00_prepare_data.py, Archetypes/00_prepare_data.py,
# Cine/00_prepare_data.py) into one script with a small per-branch lookup
# table. Delegates unchanged to Utils/prepare_features.py::main(). The source
# repo keeps its own three separate files untouched.
#
# Usage (from "Data processing/" as CWD):
#   python prepare_data.py vector
#   python prepare_data.py archetypes
#   python prepare_data.py cine
#   python prepare_data.py all          # runs the three in sequence
# =============================================================================

import sys

sys.path.insert(0, "Utils")
from prepare_features import main

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BRANCHES = {
    "vector": dict(
        config_path="Vector/config.yaml",
        out_dir="Vector/data",
    ),
    "archetypes": dict(
        config_path="Archetypes/config.yaml",
        out_dir="Archetypes/data",
    ),
    "cine": dict(
        config_path="Cine/config.yaml",
        out_dir="Cine/data",
        out_name="ies_cine_features.csv",
        group_col="cine-f 13 área",
    ),
}


def run(branch):
    if branch not in BRANCHES:
        print(f"Unknown branch '{branch}'. Valid options: {', '.join(BRANCHES)}")
        sys.exit(1)
    main(**BRANCHES[branch])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_data.py <vector|archetypes|cine|all>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "all":
        for b in BRANCHES:
            run(b)
    else:
        run(arg)
