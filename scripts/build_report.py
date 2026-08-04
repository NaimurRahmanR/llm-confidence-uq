#!/usr/bin/env python3
"""Build evidence-derived release documentation without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build verified research documentation")
    parser.add_argument("--config", default="configs/report.json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--initial-update", action="store_true")
    arguments = parser.parse_args()

    from llm_confidence_uq.reporting import preflight, run

    config_path = Path(arguments.config)
    try:
        if arguments.preflight:
            details = preflight(REPO, config_path)
            print("RELEASE DOCUMENTATION PREFLIGHT: PASS")
            print("report=" + canonical_json(details))
            print("SCOPE: read-only evidence and rendering validation; no repository files were modified.")
            return 0

        details = run(REPO, config_path, allow_initial_update=arguments.initial_update)
        print("RELEASE DOCUMENTATION BUILD: PASS")
        print("report=" + canonical_json(details))
        print("SCOPE: evidence-derived documentation only; no experiment was rerun and no result was manually entered.")
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
