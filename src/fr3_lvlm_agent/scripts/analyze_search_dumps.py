#!/usr/bin/env python3
"""
Analyze FR3 LVLM search dump JSON artifacts and emit per-pose CSV metrics.

The script computes:
- recall_proxy = accepted / evaluated
- false_positive_rate_proxy = detected_but_rejected / detected

If optional ground-truth labels are present in each JSON:
- is_target_visible (bool)
then standard recall/FPR are also computed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Stats:
    eval_count: int = 0
    detected_count: int = 0
    accepted_count: int = 0
    detected_rejected_count: int = 0
    gt_tp: int = 0
    gt_fp: int = 0
    gt_fn: int = 0
    gt_tn: int = 0
    gt_labeled_count: int = 0


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes"}:
            return True
        if v in {"0", "false", "no"}:
            return False
    return None


def _parse_args():
    p = argparse.ArgumentParser(description="Analyze FR3 LVLM search dump JSON files.")
    p.add_argument(
        "--input-dir",
        default="/tmp/fr3_lvlm_debug/search",
        help="Directory containing search dump JSON files.",
    )
    p.add_argument(
        "--glob-pattern",
        default="*.json",
        help="Glob pattern under --input-dir (default: *.json).",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path. Default: <input-dir>/search_dump_summary.csv",
    )
    p.add_argument(
        "--command-contains",
        default="",
        help="Optional case-insensitive substring filter on command.",
    )
    p.add_argument(
        "--target-color",
        default="",
        help="Optional exact target_color filter (e.g., red).",
    )
    p.add_argument(
        "--session",
        default="",
        help="Optional exact session filter.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    pattern = os.path.join(args.input_dir, "**", args.glob_pattern)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No JSON files found at: {pattern}")
        return 1

    command_filter = args.command_contains.strip().lower()
    color_filter = args.target_color.strip().lower()
    session_filter = args.session.strip()

    stats_by_key: dict[tuple[str, str, str], Stats] = defaultdict(Stats)
    reasons_by_key: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_loaded = 0
    total_used = 0

    for path in paths:
        total_loaded += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        cmd = str(rec.get("command", ""))
        cmd_l = cmd.lower()
        target_color = str(rec.get("target_color", "")).strip().lower()
        session = str(rec.get("session", "")).strip()

        if command_filter and command_filter not in cmd_l:
            continue
        if color_filter and target_color != color_filter:
            continue
        if session_filter and session != session_filter:
            continue

        total_used += 1
        phase = str(rec.get("phase", "")).strip() or "unknown"
        pose_label = str(rec.get("pose_label", "")).strip() or "unknown"
        pose_index = rec.get("pose_index")
        pose_index_str = str(pose_index) if pose_index is not None else "na"
        key = (phase, pose_label, pose_index_str)

        accepted = bool(rec.get("accepted", False))
        detected = bool(rec.get("detected", accepted))
        reject_reason = str(rec.get("reject_reason", "")).strip()

        st = stats_by_key[key]
        st.eval_count += 1
        if detected:
            st.detected_count += 1
        if accepted:
            st.accepted_count += 1
        if detected and (not accepted):
            st.detected_rejected_count += 1
            if reject_reason:
                reasons_by_key[key][reject_reason] += 1

        # Optional ground-truth field for exact metrics.
        gt_visible = _safe_bool(rec.get("is_target_visible", None))
        if gt_visible is not None:
            st.gt_labeled_count += 1
            if accepted and gt_visible:
                st.gt_tp += 1
            elif accepted and (not gt_visible):
                st.gt_fp += 1
            elif (not accepted) and gt_visible:
                st.gt_fn += 1
            else:
                st.gt_tn += 1

    if total_used == 0:
        print("No records matched the provided filters.")
        return 1

    output_csv = args.output_csv.strip() or os.path.join(args.input_dir, "search_dump_summary.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    rows = []
    for key in sorted(stats_by_key.keys()):
        phase, pose_label, pose_index_str = key
        st = stats_by_key[key]
        recall_proxy = float(st.accepted_count) / float(st.eval_count) if st.eval_count > 0 else 0.0
        fpr_proxy = (
            float(st.detected_rejected_count) / float(st.detected_count)
            if st.detected_count > 0
            else 0.0
        )

        recall_gt = ""
        fpr_gt = ""
        if st.gt_labeled_count > 0:
            denom_rec = st.gt_tp + st.gt_fn
            denom_fpr = st.gt_fp + st.gt_tn
            recall_gt = f"{(float(st.gt_tp) / float(denom_rec)):.6f}" if denom_rec > 0 else ""
            fpr_gt = f"{(float(st.gt_fp) / float(denom_fpr)):.6f}" if denom_fpr > 0 else ""

        reason_counts = reasons_by_key.get(key, {})
        top_reasons = ",".join(
            f"{r}:{c}" for r, c in sorted(reason_counts.items(), key=lambda rc: (-rc[1], rc[0]))[:6]
        )

        rows.append(
            {
                "phase": phase,
                "pose_label": pose_label,
                "pose_index": pose_index_str,
                "eval_count": st.eval_count,
                "detected_count": st.detected_count,
                "accepted_count": st.accepted_count,
                "detected_rejected_count": st.detected_rejected_count,
                "recall_proxy": f"{recall_proxy:.6f}",
                "false_positive_rate_proxy": f"{fpr_proxy:.6f}",
                "gt_labeled_count": st.gt_labeled_count,
                "recall_gt": recall_gt,
                "fpr_gt": fpr_gt,
                "top_reject_reasons": top_reasons,
            }
        )

    fieldnames = [
        "phase",
        "pose_label",
        "pose_index",
        "eval_count",
        "detected_count",
        "accepted_count",
        "detected_rejected_count",
        "recall_proxy",
        "false_positive_rate_proxy",
        "gt_labeled_count",
        "recall_gt",
        "fpr_gt",
        "top_reject_reasons",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Loaded {total_loaded} JSON files, analyzed {total_used} records.")
    print(f"Wrote CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
