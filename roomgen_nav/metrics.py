"""PointNav metric aggregation.

The per-episode numbers come from :meth:`roomgen_nav.env.PointNavEnv.evaluate`;
this module only groups them.  Definitions match the Habitat PointNav
convention:

* **success** -- final 3D distance to the goal <= ``success_radius`` *and* the
  agent is on the goal's floor (a goal above the head does not count);
* **SPL** -- ``shortest / max(path_length, shortest)`` for successful episodes,
  zero otherwise;
* **collision** -- a ``forward`` action whose realised movement was less than
  20% of the commanded step;
* **timeout** -- the episode hit ``max_steps``;
* **final distance** -- 3D distance at the end of the stored action sequence.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _group_summary(results: Sequence[dict]) -> dict:
    count = len(results)
    successes = [result for result in results if result.get("success")]
    timeouts = sum(1 for result in results if result.get("timeout"))
    return {
        "episodes": count,
        "successes": len(successes),
        "success_rate": round(len(successes) / count, 4) if count else 0.0,
        # SPL over successful episodes (kept for the historical report shape).
        "spl": round(_mean([float(r.get("spl", 0.0)) for r in successes]), 4),
        # Standard SPL over every episode in the group; failures score zero.
        "spl_all": round(_mean([float(r.get("spl", 0.0)) for r in results]), 4),
        "collisions": int(sum(int(r.get("collisions", 0)) for r in results)),
        "timeouts": int(timeouts),
        "timeout_rate": round(timeouts / count, 4) if count else 0.0,
        "final_distance_mean": round(_mean([float(r.get("final_distance", 0.0))
                                            for r in results]), 4),
        "steps_mean": round(_mean([float(r.get("steps", 0)) for r in results]), 2),
        "path_length_mean": round(_mean([float(r.get("path_length", 0.0))
                                         for r in results]), 4),
    }


def summarize_results(results: Sequence[dict]) -> dict:
    count = len(results)
    successes = [result for result in results if result.get("success")]
    failure_reasons = Counter(result.get("failure_reason") or "unknown"
                              for result in results if not result.get("success"))
    cross = [result for result in results if result.get("cross_floor")]
    same = [result for result in results if not result.get("cross_floor")]
    summary = _group_summary(results)
    summary.update({
        "failures": count - len(successes),
        "failure_rate": round((count - len(successes)) / count, 4) if count else 0.0,
        "collisions_per_episode": round(
            sum(int(r.get("collisions", 0)) for r in results) / count, 4) if count else 0.0,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "cross_floor": _group_summary(cross),
        "same_floor": _group_summary(same),
    })
    return summary


def summarize_by_scene(scene_results: Dict[str, Sequence[dict]]) -> dict:
    out = {}
    for scene_id, results in sorted(scene_results.items()):
        summary = summarize_results(results)
        out[scene_id] = summary
    return out


def aggregate_run(run: dict) -> dict:
    """Add the all-scene summary to a run dict and return it."""
    all_results: List[dict] = []
    for scene in run.get("scenes", []):
        all_results.extend(scene.get("results", []))
    run["summary"] = summarize_results(all_results)
    run["scene_summaries"] = {scene["scene_id"]: scene.get("summary", {})
                              for scene in run.get("scenes", [])}
    return run


def write_run(path: str, run: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=2)


def format_summary(summary: dict, heading: Optional[str] = None) -> str:
    lines = []
    if heading:
        lines.append(heading)
    lines.extend([
        f"episodes        : {summary.get('episodes', 0)}",
        f"success rate    : {summary.get('success_rate', 0.0):.3f}",
        f"SPL (successes) : {summary.get('spl', 0.0):.3f}",
        f"SPL (all)       : {summary.get('spl_all', summary.get('spl', 0.0)):.3f}",
        f"collisions      : {summary.get('collisions', 0)} "
        f"({summary.get('collisions_per_episode', 0.0):.3f}/episode)",
        f"timeout rate    : {summary.get('timeout_rate', 0.0):.3f}",
        f"final distance  : {summary.get('final_distance_mean', 0.0):.3f} m",
        f"failure reasons : {summary.get('failure_reasons', {})}",
    ])
    cross = summary.get("cross_floor", {})
    same = summary.get("same_floor", {})
    lines.append(f"cross-floor     : {cross.get('episodes', 0)} episodes, "
                 f"success {cross.get('success_rate', 0.0):.3f}, "
                 f"SPL(all) {cross.get('spl_all', 0.0):.3f}")
    lines.append(f"same-floor      : {same.get('episodes', 0)} episodes, "
                 f"success {same.get('success_rate', 0.0):.3f}, "
                 f"SPL(all) {same.get('spl_all', 0.0):.3f}")
    return "\n".join(lines)
