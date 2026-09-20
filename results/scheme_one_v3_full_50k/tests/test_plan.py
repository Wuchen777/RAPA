#!/usr/bin/env python3
"""Static protocol checks for the full V3 experiment."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_full.py"
spec = importlib.util.spec_from_file_location("v3_full_runner_test", RUNNER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    patches = module.patch_jobs()
    evaluations = module.eval_jobs()
    assert len(module.smoke_jobs()) == 3
    assert len(patches) == 60
    assert len({job.key for job in patches}) == 60
    assert len(evaluations) == 135
    assert len({job.key for job in evaluations}) == 135
    assert {table: sum(job.table == table for job in evaluations) for table in ("table2", "table3", "table4", "table6")} == {
        "table2": 12, "table3": 24, "table4": 27, "table6": 72,
    }
    assert all("--num_trials_per_task" in job.command and "50" in job.command for job in evaluations)
    assert all(job.seed in (7, 21, 42) for job in patches + evaluations)

    original_is_done = module.is_done
    module.is_done = lambda job: job.kind == "smoke"
    phase, workload, concurrency = module.phase()
    module.is_done = original_is_done
    assert phase == "patches_and_evaluations"
    assert len(workload) == 195
    assert concurrency == 8

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        primary = root / "primary.pt"
        wrist = root / "wrist.pt"
        dependency_job = module.Job(
            "eval", "table6", "openvla_oft", "libero_spatial", 7, [], root / "eval.log",
            primary, wrist,
        )
        assert not module.dependencies_ready(dependency_job)
        primary.touch(); module.marker(primary).touch()
        assert not module.dependencies_ready(dependency_job)
        wrist.touch(); module.marker(wrist).touch()
        assert module.dependencies_ready(dependency_job)
    print("PASS full V3 protocol: 60 patch jobs, 135 evaluation jobs")


if __name__ == "__main__":
    main()
