#!/usr/bin/env python
"""Evaluate ``tools/validate_blender_agent_p0.py``'s report with a resolution rule.

Both ``tests/test_front3d_replay.sh`` (low-resolution iteration profile) and
``scripts/front3d_replay.sh`` (any ``PANOWORLD_RES``) use this so that a run at a
non-canonical size is not reported as a failure: the validator's own
``canonical_resolution`` check is the *only* one allowed to fail below
2048x1024, and at the canonical size every check must pass.

    python tests/eval_validation_report.py --report <report.json> --resolution 1024x512
"""

from __future__ import annotations

import argparse
import json
import sys

CANONICAL = "2048x1024"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", required=True, help="WxH the build was rendered at")
    args = parser.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    checks = report.get("checks", [])
    failed = [check["name"] for check in checks if not check["pass"]]
    allowed = [] if args.resolution == CANONICAL else ["canonical_resolution"]
    unexpected = [name for name in failed if name not in allowed]

    for check in checks:
        print(f"[validate] {'PASS' if check['pass'] else 'FAIL'}  {check['name']}")
    if args.resolution != CANONICAL:
        print(f"[validate] resolution {args.resolution}: canonical_resolution is expected to fail "
              f"below the canonical size ({CANONICAL}) -- the depth/pose checks are the ones that matter")
    if unexpected:
        print(f"[validate] UNEXPECTED FAILURES: {unexpected}", file=sys.stderr)
        return 1
    skipped = len(allowed)
    print(f"[validate] overall: PASS ({len(checks) - len(failed)}/{len(checks)} checks"
          + (f", {skipped} skipped by design at this resolution)" if skipped else ")"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
