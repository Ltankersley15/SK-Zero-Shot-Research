#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from PIL import Image

from fr3_lvlm_agent.reasoning.local_vl_verifier import LocalVLCandidate, LocalVisionVerifier
from fr3_lvlm_agent.reasoning.ollama_client import OllamaClient


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local LVLM verifier on saved Isaac artifacts.")
    parser.add_argument("--manifest", required=True, help="Path to JSON manifest with image/candidate samples.")
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--max-crop-px", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    client = OllamaClient(model=args.model, base_url=args.base_url, timeout=args.timeout_sec)
    verifier = LocalVisionVerifier(
        client=client,
        max_crop_px=args.max_crop_px,
        default_timeout_sec=args.timeout_sec,
    )

    samples = list(manifest.get("samples", []))
    if not samples:
        raise SystemExit("manifest did not contain any samples")

    results = []
    latencies = []
    correct = 0
    for sample in samples:
        image = Image.open(Path(sample["image"]).expanduser()).convert("RGB")
        candidates = [
            LocalVLCandidate(
                candidate_id=str(candidate["candidate_id"]),
                label=str(candidate.get("label", candidate["candidate_id"])),
                bbox_xyxy=tuple(int(v) for v in candidate["bbox_xyxy"]) if candidate.get("bbox_xyxy") else None,
                center_uv=tuple(int(v) for v in candidate["center_uv"]) if candidate.get("center_uv") else None,
                source=str(candidate.get("source", "")),
            )
            for candidate in sample.get("candidates", [])
        ]
        decision = verifier.choose_named_container(
            image=image,
            target_text=str(sample["target"]),
            candidates=candidates,
            confidence_threshold=float(args.threshold),
        )
        expected = sample.get("expected_candidate_id")
        is_correct = expected is not None and decision.selected_candidate_id == str(expected)
        correct += int(is_correct)
        latencies.append(float(decision.latency_sec))
        results.append(
            {
                "image": str(sample["image"]),
                "target": str(sample["target"]),
                "expected_candidate_id": expected,
                "selected_candidate_id": decision.selected_candidate_id,
                "status": decision.status,
                "confidence": decision.confidence,
                "latency_sec": decision.latency_sec,
                "correct": is_correct,
            }
        )

    summary = {
        "model": args.model,
        "num_samples": len(results),
        "accuracy": (correct / len(results)) if results else 0.0,
        "median_latency_sec": median(latencies) if latencies else 0.0,
        "p95_latency_sec": sorted(latencies)[max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))] if latencies else 0.0,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
