"""Replay saved Day 6 lifting cases and fail on geometry drift."""

import argparse
import json
from pathlib import Path

from qmapnav.mapping.lifting_regression import replay_lifting_regression_case


def main() -> None:
    """Replay every direct child containing a Day 6 manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    arguments = parser.parse_args()
    cases = sorted(
        path.parent for path in arguments.root.glob('*/manifest.json')
    )
    if not cases:
        raise SystemExit(f'no Day 6 cases found below {arguments.root}')
    payload = {}
    failed = False
    for case in cases:
        metrics = replay_lifting_regression_case(case)
        payload[case.name] = {
            'status_matches': metrics.status_matches,
            'point_indices_match': metrics.point_indices_match,
            'centre_error_m': metrics.centre_error_m,
            'dimension_error_m': metrics.dimension_error_m,
            'yaw_error_rad': metrics.yaw_error_rad,
            'checksum_valid': metrics.checksum_valid,
            'passed': metrics.passed,
        }
        failed = failed or not metrics.passed
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
