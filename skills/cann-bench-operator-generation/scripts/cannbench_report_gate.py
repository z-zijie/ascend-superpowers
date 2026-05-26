#!/usr/bin/env python3
"""Validate that a CANN-Bench JSON report is eligible for delivery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _case_accuracy_passed(case: dict[str, Any]) -> bool:
    accuracy = case.get("accuracy")
    if accuracy is None:
        return case.get("status") == "success"
    return bool(accuracy.get("passed"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail unless a CANN-Bench report passes the delivery gate."
    )
    parser.add_argument("report", type=Path, help="Path to CANN-Bench JSON report")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Required overall score. Omit when the goal has no score target.",
    )
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text())
    except Exception as exc:  # pragma: no cover - surfaced in CLI output
        print(f"FAIL: cannot read report {args.report}: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []

    total_cases = int(report.get("total_cases") or 0)
    passed_cases = int(report.get("passed_cases") or 0)
    failed_cases = int(report.get("failed_cases") or 0)
    overall_score = report.get("overall_score")

    if total_cases <= 0:
        failures.append("report has no cases")
    if passed_cases != total_cases or failed_cases != 0:
        failures.append(
            f"case summary is not all-pass: passed={passed_cases}, "
            f"failed={failed_cases}, total={total_cases}"
        )

    if args.min_score is not None:
        if overall_score is None:
            failures.append("overall_score is missing")
        elif float(overall_score) < args.min_score:
            failures.append(
                f"overall_score {float(overall_score):.2f} < required {args.min_score:.2f}"
            )

    for operator in report.get("operators") or []:
        operator_name = operator.get("operator", "<unknown>")
        op_total = int(operator.get("total_cases") or 0)
        op_passed = int(operator.get("passed_cases") or 0)
        op_failed = int(operator.get("failed_cases") or 0)
        if op_total <= 0:
            failures.append(f"{operator_name}: operator has no cases")
        if op_passed != op_total or op_failed != 0:
            failures.append(
                f"{operator_name}: operator summary is not all-pass "
                f"(passed={op_passed}, failed={op_failed}, total={op_total})"
            )

        for case in operator.get("cases") or []:
            case_id = case.get("case_id", "<unknown>")
            status = case.get("status")
            if status != "success":
                error = case.get("error_msg") or "<no error_msg>"
                failures.append(f"{operator_name} {case_id}: status={status}: {error}")
                continue
            if not _case_accuracy_passed(case):
                failures.append(f"{operator_name} {case_id}: accuracy did not pass")

    if failures:
        print("FAIL: CANN-Bench delivery gate failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    score_text = "n/a" if overall_score is None else f"{float(overall_score):.2f}"
    print(
        "PASS: CANN-Bench delivery gate passed "
        f"(cases={passed_cases}/{total_cases}, score={score_text})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
