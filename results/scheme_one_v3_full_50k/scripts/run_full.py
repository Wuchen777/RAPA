#!/usr/bin/env python3
"""Run the complete three-model TC-TD V3 experiment with dynamic GPU scheduling."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts import gpu_reservations

RESULTS_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = RESULTS_ROOT / "code"
LIBERO_CONFIG_ROOT = RESULTS_ROOT / "libero_config"
VLA_PYTHON = Path(os.environ.get("VLA_PYTHON", "python"))
OFT_PYTHON = Path(os.environ.get("OFT_PYTHON", "python"))

MODELS = ("openvla", "openvla_oft", "pi0")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TRANSFER_TARGETS = ("libero_object", "libero_goal", "libero_10")
SEEDS = (7, 21, 42)
DATASET = {suite: f"{suite}_no_noops" for suite in SUITES}
OPENVLA_CKPT = {suite: f"openvla/openvla-7b-finetuned-{suite.replace('_', '-')}" for suite in SUITES}
OFT_CKPT = {suite: f"moojink/openvla-7b-oft-finetuned-{suite.replace('_', '-')}" for suite in SUITES}
PI0_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pi0_libero"
PI0_POLICY_CONFIG = "pi0_libero"
FAST_TOKENIZER = PROJECT_ROOT / "checkpoints" / "hf" / "physical-intelligence" / "fast"

PATCH_STEPS = 50000
BATCH_PROBE_CANDIDATES = (16, 20, 24, 28, 32, 36, 40, 48, 56, 60, 64)
BATCH_PROBE_START = 32
BATCH_SELECTION_PATH = RESULTS_ROOT / "batch_size_selection.json"
BATCH_SIZE = BATCH_PROBE_START
NUM_VIEWS = 4
RATIO = 0.05
ALPHA = 0.8
STEP_SIZE = 2 / 255
RELATION_TEMPERATURE = 0.1
EPISODES_PER_TASK = 50
EXPECTED_EPISODES = 500
PATCH_CONCURRENCY = 8
EVAL_CONCURRENCY = 8
POLL_SECONDS = 10
MIN_MODEL_LAUNCH_FREE_MIB = 15000
GPU_LAUNCH_LOCK_PATH = PROJECT_ROOT / "results" / ".gpu_launch.lock"
OOM_MARKERS = (
    "CUDA out of memory", "OutOfMemoryError", "CUDA out of", "CUBLAS_STATUS_ALLOC_FAILED",
    "RESOURCE_EXHAUSTED", "out of memory",
)


@dataclass(frozen=True)
class Job:
    kind: str
    table: str
    model: str
    suite: str
    seed: int
    command: list[str]
    log_path: Path
    patch_primary: Path
    patch_wrist: Path | None = None
    source_model: str | None = None
    source_suite: str | None = None
    target_model: str | None = None
    required_free_mib: int = MIN_MODEL_LAUNCH_FREE_MIB

    @property
    def key(self) -> str:
        view = self.patch_primary.parent.name.split("-")[0] if self.kind in {"patch", "smoke"} else ""
        return ":".join(str(value or "") for value in (
            self.kind, self.table, self.model, self.source_model, self.target_model,
            self.source_suite, self.suite, self.seed, view,
        ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "full", "aggregate"), default="plan")
    return parser.parse_args()


def views(model: str) -> tuple[str, ...]:
    return ("primary",) if model == "openvla" else ("primary", "wrist")


def patch_path(model: str, suite: str, seed: int, view: str) -> Path:
    suffix = "npy" if model == "pi0" else "pt"
    return RESULTS_ROOT / "patches" / model / suite / f"seed_{seed}" / f"{view}-{RATIO}" / f"perturbation.{suffix}"


def marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".complete")


def patch_generator(model: str) -> tuple[Path, Path]:
    if model == "openvla":
        return VLA_PYTHON, CODE_ROOT / "generate_patch_openvla.py"
    if model == "openvla_oft":
        return OFT_PYTHON, CODE_ROOT / "generate_patch_openvla_oft.py"
    return VLA_PYTHON, CODE_ROOT / "generate_patch_pi0.py"


def checkpoint(model: str, suite: str) -> str:
    if model == "openvla":
        return OPENVLA_CKPT[suite]
    if model == "openvla_oft":
        return OFT_CKPT[suite]
    return str(PI0_CHECKPOINT)


def make_patch_job(
    model: str,
    suite: str,
    seed: int,
    view: str,
    steps: int,
    smoke: bool = False,
    batch_size: int | None = None,
    probe: bool = False,
) -> Job:
    effective_batch_size = BATCH_SIZE if batch_size is None else batch_size
    python, script = patch_generator(model)
    if probe:
        root = RESULTS_ROOT / "batch_probe" / f"batch_{effective_batch_size}" / model / suite / f"seed_{seed}" / view
    else:
        root = RESULTS_ROOT / ("smoke" if smoke else "patches") / model / suite / f"seed_{seed}" / view
    output = root.with_name(f"{view}-{RATIO}") / ("perturbation.npy" if model == "pi0" else "perturbation.pt")
    command = [
        str(python), str(script),
        "--vla_path", checkpoint(model, suite),
        "--data_root_dir", str(PROJECT_ROOT / "dataset" / "modified_libero_rlds"),
        "--dataset_name", DATASET[suite],
        "--batch_size", str(effective_batch_size),
        "--max_steps", str(steps),
        "--perturbation_ratio", str(RATIO),
        "--alpha", str(ALPHA),
        "--step_size", str(STEP_SIZE),
        "--num_views", str(NUM_VIEWS),
        "--relation_temperature", str(RELATION_TEMPERATURE),
        "--save_path", str(root),
        "--camera_view", view,
        "--seed", str(seed),
        "--use_wandb", "False",
    ]
    if model == "pi0":
        command += ["--policy_config_name", PI0_POLICY_CONFIG, "--cycle_dataloader", "True"]
    table = f"batch_probe_{effective_batch_size}" if probe else ("smoke" if smoke else "patch_generation")
    log = RESULTS_ROOT / "logs" / f"{table}_{model}_{suite}_seed{seed}_{view}.log"
    return Job("smoke" if smoke else "patch", table, model, suite, seed, command, log, output)


def smoke_jobs() -> list[Job]:
    return [make_patch_job(model, "libero_spatial", 7, "primary", 1, smoke=True) for model in MODELS]


def patch_jobs() -> list[Job]:
    return [
        make_patch_job(model, suite, seed, view, PATCH_STEPS)
        for model in MODELS for suite in SUITES for seed in SEEDS for view in views(model)
    ]


def eval_log(table: str, source_model: str, target_model: str, source_suite: str, suite: str, seed: int) -> Path:
    return RESULTS_ROOT / "logs" / f"{table}_{source_model}_to_{target_model}_{source_suite}_to_{suite}_seed{seed}.log"


def eval_command(target: str, suite: str, seed: int, primary: Path, wrist: Path | None, note: str):
    common = [
        "--task_suite_name", suite, "--num_trials_per_task", str(EPISODES_PER_TASK),
        "--max_tasks", "0", "--eval_max_steps", "0", "--seed", str(seed),
        "--local_log_dir", str(RESULTS_ROOT / "eval_logs"), "--run_id_note", note,
        "--patch_attack", "True",
    ]
    if target == "openvla":
        return [
            str(VLA_PYTHON), "eval/simulation/Libero/openvla.py",
            "--pretrained_checkpoint", OPENVLA_CKPT[suite], *common,
            "--perturbation_path", str(primary),
        ], 15000
    if wrist is None:
        wrist = primary
    if target == "openvla_oft":
        return [
            str(OFT_PYTHON), "eval/simulation/Libero/openvla_oft.py",
            "--pretrained_checkpoint", OFT_CKPT[suite], "--use_wandb", "False", *common,
            "--perturbation_primary_path", str(primary), "--perturbation_wrist_path", str(wrist),
        ], 15000
    return [
        str(VLA_PYTHON), "eval/simulation/Libero/pi0.py",
        "--checkpoint_path", str(PI0_CHECKPOINT), "--policy_config_name", PI0_POLICY_CONFIG, *common,
        "--perturbation_primary_path", str(primary), "--perturbation_wrist_path", str(wrist),
    ], 16000


def make_eval_job(table: str, source_model: str, target_model: str, source_suite: str, suite: str, seed: int) -> Job:
    primary = patch_path(source_model, source_suite, seed, "primary")
    wrist = patch_path(source_model, source_suite, seed, "wrist") if source_model != "openvla" else None
    note = f"v3-full-{table}-{source_model}-to-{target_model}-{source_suite}-to-{suite}-seed{seed}"
    command, required = eval_command(target_model, suite, seed, primary, wrist, note)
    return Job(
        "eval", table, target_model, suite, seed, command,
        eval_log(table, source_model, target_model, source_suite, suite, seed),
        primary, wrist, source_model, source_suite, target_model, required,
    )


def eval_jobs() -> list[Job]:
    result = []
    for suite in SUITES:
        for seed in SEEDS:
            result.append(make_eval_job("table2", "openvla", "openvla", suite, suite, seed))
    for model in ("openvla_oft", "pi0"):
        for suite in SUITES:
            for seed in SEEDS:
                result.append(make_eval_job("table3", model, model, suite, suite, seed))
    for model in MODELS:
        for suite in TRANSFER_TARGETS:
            for seed in SEEDS:
                result.append(make_eval_job("table4", model, model, "libero_spatial", suite, seed))
    for source in MODELS:
        for target in MODELS:
            if source == target:
                continue
            for suite in SUITES:
                for seed in SEEDS:
                    result.append(make_eval_job("table6", source, target, suite, suite, seed))
    return result


def ensure_libero_config() -> Path:
    LIBERO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    benchmark = PROJECT_ROOT / "LIBERO" / "libero" / "libero"
    (LIBERO_CONFIG_ROOT / "config.yaml").write_text("\n".join([
        f"assets: {benchmark / 'assets'}", f"bddl_files: {benchmark / 'bddl_files'}",
        f"benchmark_root: {benchmark}", f"datasets: {PROJECT_ROOT / 'LIBERO' / 'libero' / 'datasets'}",
        f"init_states: {benchmark / 'init_files'}", "",
    ]), encoding="utf-8")
    return LIBERO_CONFIG_ROOT


def parse_eval(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last):" in content or "Episode error:" in content:
        return None
    if any(marker.lower() in content.lower() for marker in OOM_MARKERS):
        return None
    episodes = successes = None
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if match := re.search(r"Total episodes:\s*(\d+)", line): episodes = int(match.group(1))
        if match := re.search(r"Total successes:\s*(\d+)", line): successes = int(match.group(1))
        if match := re.search(r"# episodes completed so far:\s*(\d+)", line): episodes = int(match.group(1))
        elif "# episodes completed so far:" in line:
            for candidate in lines[index + 1:index + 4]:
                if match := re.search(r"^\s*(\d+)\s*$", candidate): episodes = int(match.group(1)); break
        if match := re.search(r"# successes:\s*(\d+)", line): successes = int(match.group(1))
    return (episodes, successes) if episodes is not None and successes is not None else None


def is_done(job: Job) -> bool:
    if job.kind == "smoke":
        validation = RESULTS_ROOT / "smoke" / f"{job.model}.validated"
        if not validation.exists():
            return False
        try:
            payload = json.loads(validation.read_text(encoding="utf-8"))
            return payload.get("batch_size") == BATCH_SIZE
        except (OSError, json.JSONDecodeError):
            return False
    if job.kind == "patch":
        return job.patch_primary.exists() and marker(job.patch_primary).exists()
    parsed = parse_eval(job.log_path)
    return parsed is not None and parsed[0] >= EXPECTED_EPISODES


def dependencies_ready(job: Job) -> bool:
    if job.kind != "eval":
        return True
    required = [job.patch_primary]
    if job.patch_wrist is not None:
        required.append(job.patch_wrist)
    return all(path.exists() and marker(path).exists() for path in required)


def environment(job: Job, gpu: int, config: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
        "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "offline", "SAVE_ROLLOUTS": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OPENPI_FAST_TOKENIZER_PATH": str(FAST_TOKENIZER), "LIBERO_CONFIG_PATH": str(config),
        "PI0_CHECKPOINT_PATH": str(PI0_CHECKPOINT), "PI0_POLICY_CONFIG_NAME": PI0_POLICY_CONFIG,
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"), "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
        "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "4"),
    })
    roots = [CODE_ROOT, PROJECT_ROOT, PROJECT_ROOT / "LIBERO", PROJECT_ROOT / "dlimp_openvla"]
    if job.model == "openvla_oft": roots.insert(1, PROJECT_ROOT / "openvla_oft")
    elif job.model == "pi0": roots[1:1] = [
        PROJECT_ROOT / "openpi" / "src", PROJECT_ROOT / "openpi" / "packages" / "openpi-client" / "src",
        PROJECT_ROOT / "lerobot", PROJECT_ROOT / "openvla",
    ]
    else: roots.insert(1, PROJECT_ROOT / "openvla")
    if env.get("PYTHONPATH"): roots.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(root) for root in roots)
    return env


def gpu_snapshot():
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    app_query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    busy = defaultdict(list)
    for line in app_query.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[1].isdigit(): busy[fields[0]].append(int(fields[1]))
    result = []
    for line in gpu_query.stdout.splitlines():
        index, uuid, free = [part.strip() for part in line.split(",")]
        result.append({"index": int(index), "uuid": uuid, "free": int(free), "pids": busy[uuid]})
    return result


def gpu_launch_lock():
    lock = GPU_LAUNCH_LOCK_PATH.open("a", encoding="utf-8")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def atomic_state(payload):
    path = RESULTS_ROOT / "state.json"; temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def smoke_metrics(job: Job) -> dict | None:
    metrics = job.patch_primary.parent / "metrics.jsonl"
    if not job.patch_primary.exists() or not metrics.exists(): return None
    try: record = json.loads(metrics.read_text(encoding="utf-8").splitlines()[-1])
    except (OSError, json.JSONDecodeError, IndexError): return None
    numeric = [value for value in record.values() if isinstance(value, (int, float))]
    required = (
        all(math.isfinite(float(value)) for value in numeric)
        and record.get("grad_norm_edpa", 0) > 0
        and record.get("grad_norm_auxiliary_projected", 0) > 0
        and record.get("token_relation_js_raw", 0) > 0
        and record.get("auxiliary_edpa_dot_after", -1) >= -1e-6
        and record.get("update_edpa_dot", -1) >= 0
    )
    return record if required else None


def validate_smoke(job: Job) -> bool:
    record = smoke_metrics(job)
    validation = RESULTS_ROOT / "smoke" / f"{job.model}.validated"
    validation.parent.mkdir(parents=True, exist_ok=True)
    if record is not None:
        payload = {"batch_size": BATCH_SIZE, "metrics": record}
        validation.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
        return True
    return False


def selected_batch_size() -> int | None:
    if not BATCH_SELECTION_PATH.exists():
        return None
    try:
        payload = json.loads(BATCH_SELECTION_PATH.read_text(encoding="utf-8"))
        selected = int(payload["selected_batch_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return selected if selected in BATCH_PROBE_CANDIDATES else None


def write_batch_selection(selected: int, tested: dict[int, bool]) -> None:
    payload = {
        "status": "complete",
        "time": time.strftime("%F %T"),
        "selected_batch_size": selected,
        "shared_across_models": list(MODELS),
        "probe_start": BATCH_PROBE_START,
        "candidates": list(BATCH_PROBE_CANDIDATES),
        "tested": {str(batch): passed for batch, passed in sorted(tested.items())},
        "criterion": "one complete optimization step with finite V3 metrics on every model",
    }
    temporary = BATCH_SELECTION_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, BATCH_SELECTION_PATH)


def probe_batch_candidate(batch_size: int, config: Path) -> bool:
    jobs = [
        make_patch_job(model, "libero_spatial", 7, "primary", 1, smoke=True, batch_size=batch_size, probe=True)
        for model in MODELS
    ]
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen, Job, object, int]] = {}
    outcomes: dict[str, bool] = {}
    while pending or running:
        for key, (process, job, log_file, gpu) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log_file.close()
            del running[key]
            content = job.log_path.read_text(encoding="utf-8", errors="replace") if job.log_path.exists() else ""
            oom = any(value.lower() in content.lower() for value in OOM_MARKERS)
            valid = code == 0 and smoke_metrics(job) is not None
            if not valid and not oom:
                raise RuntimeError(
                    f"Batch probe failed for {job.model} at batch={batch_size} for a non-OOM reason; "
                    f"inspect {job.log_path}"
                )
            outcomes[job.model] = valid
            print(f"[batch-probe] batch={batch_size} model={job.model} gpu={gpu} valid={valid} oom={oom}", flush=True)

        reserved_gpus = {value[3] for value in running.values()} | gpu_reservations.reserved_gpus()
        available = [
            gpu for gpu in gpu_snapshot()
            if gpu["index"] not in reserved_gpus and gpu["free"] >= MIN_MODEL_LAUNCH_FREE_MIB
        ]
        available.sort(key=lambda gpu: gpu["free"], reverse=True)
        while pending and available:
            job = pending[0]
            gpu = available[0]
            launch_lock = gpu_launch_lock()
            try:
                latest = next((item for item in gpu_snapshot() if item["index"] == gpu["index"]), None)
                if (
                    latest is None
                    or gpu["index"] in gpu_reservations.reserved_gpus()
                    or latest["pids"]
                    or latest["free"] < MIN_MODEL_LAUNCH_FREE_MIB
                ):
                    available.pop(0)
                    continue
                pending.pop(0)
                available.pop(0)
                if job.patch_primary.parent.exists():
                    shutil.rmtree(job.patch_primary.parent)
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                if job.log_path.exists() and job.log_path.stat().st_size:
                    job.log_path.replace(job.log_path.with_suffix(f".previous-{int(time.time())}.log"))
                log_file = job.log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    job.command,
                    cwd=PROJECT_ROOT,
                    env=environment(job, gpu["index"], config),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                gpu_reservations.reserve(gpu["index"], process.pid, "v3_batch_probe", job.key)
            finally:
                fcntl.flock(launch_lock, fcntl.LOCK_UN)
                launch_lock.close()
            running[job.key] = (process, job, log_file, gpu["index"])
            print(f"[batch-probe-start] batch={batch_size} model={job.model} gpu={gpu['index']}", flush=True)
        atomic_state({
            "status": "batch_probe",
            "time": time.strftime("%F %T"),
            "batch_size": batch_size,
            "completed_models": sorted(outcomes),
            "running": [{"model": value[1].model, "gpu": value[3], "pid": value[0].pid} for value in running.values()],
        })
        if pending or running:
            time.sleep(POLL_SECONDS)
    return all(outcomes.get(model, False) for model in MODELS)


def resolve_batch_size(config: Path) -> int:
    existing = selected_batch_size()
    if existing is not None:
        print(f"[batch-probe] reusing selected batch_size={existing}", flush=True)
        return existing

    candidates = list(BATCH_PROBE_CANDIDATES)
    start_index = candidates.index(BATCH_PROBE_START)
    tested: dict[int, bool] = {}

    def test(index: int) -> bool:
        batch = candidates[index]
        if batch not in tested:
            tested[batch] = probe_batch_candidate(batch, config)
        return tested[batch]

    if test(start_index):
        low, high = start_index, len(candidates)
        if test(len(candidates) - 1):
            low = len(candidates) - 1
        else:
            high = len(candidates) - 1
            while high - low > 1:
                middle = (low + high) // 2
                if test(middle):
                    low = middle
                else:
                    high = middle
    else:
        low = -1
        for index in range(start_index - 1, -1, -1):
            if test(index):
                low = index
                break
    if low < 0:
        raise RuntimeError("No batch size candidate, including 16, passed all three V3 model smoke tests")
    selected = candidates[low]
    write_batch_selection(selected, tested)
    print(f"[batch-probe-selected] batch_size={selected}", flush=True)
    return selected


def phase():
    smoke = smoke_jobs()
    if any(not is_done(job) for job in smoke):
        return "smoke", smoke, len(smoke)
    workload = patch_jobs() + eval_jobs()
    if any(not is_done(job) for job in workload):
        return "patches_and_evaluations", workload, max(PATCH_CONCURRENCY, EVAL_CONCURRENCY)
    return "complete", [], 0


def run_full() -> int:
    global BATCH_SIZE
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True); (RESULTS_ROOT / "logs").mkdir(exist_ok=True); (RESULTS_ROOT / "eval_logs").mkdir(exist_ok=True)
    config = ensure_libero_config()
    BATCH_SIZE = resolve_batch_size(config)
    attempts = {}; cooldown = {}; running = {}
    with (RESULTS_ROOT / "jobs.jsonl").open("a", encoding="utf-8") as metadata:
        while True:
            for key, (process, job, log_file, gpu) in list(running.items()):
                code = process.poll()
                if code is None: continue
                log_file.close(); del running[key]
                successful = code == 0
                if successful and job.kind == "smoke": successful = validate_smoke(job)
                elif successful and job.kind == "patch" and job.patch_primary.exists():
                    marker(job.patch_primary).write_text(json.dumps({"time": time.strftime("%F %T"), "command": job.command}) + "\n", encoding="utf-8")
                    successful = True
                elif successful and job.kind == "eval": successful = is_done(job)
                if successful: print(f"[complete] gpu={gpu} {key}", flush=True)
                else:
                    content = job.log_path.read_text(encoding="utf-8", errors="replace") if job.log_path.exists() else ""
                    oom = any(value.lower() in content.lower() for value in OOM_MARKERS)
                    print(f"[retry] gpu={gpu} code={code} oom={oom} {key}", flush=True); cooldown[gpu] = time.time() + (120 if oom else 60)
                    if not oom and attempts[key] >= 3:
                        atomic_state({"status": "failed", "time": time.strftime("%F %T"), "job": key}); return 1

            phase_name, planned, concurrency = phase()
            if phase_name == "complete" and not running:
                aggregate(); atomic_state({"status": "complete", "time": time.strftime("%F %T")}); return 0
            pending = [
                job for job in planned
                if not is_done(job) and job.key not in running and dependencies_ready(job)
            ]
            reserved_gpus = {value[3] for value in running.values()} | gpu_reservations.reserved_gpus()
            available = [
                gpu for gpu in gpu_snapshot()
                if gpu["index"] not in reserved_gpus
                and time.time() >= cooldown.get(gpu["index"], 0)
            ]
            available.sort(key=lambda gpu: gpu["free"], reverse=True)
            while pending and available and len(running) < concurrency:
                selection = next(
                    (
                        (index, job, gpu)
                        for index, job in enumerate(pending)
                        for gpu in available
                        if gpu["free"] >= job.required_free_mib
                    ),
                    None,
                )
                if selection is None:
                    break
                pending_index, job, gpu_info = selection
                launch_lock = gpu_launch_lock()
                try:
                    latest = next((gpu for gpu in gpu_snapshot() if gpu["index"] == gpu_info["index"]), None)
                    if (
                        latest is None
                        or gpu_info["index"] in gpu_reservations.reserved_gpus()
                        or latest["free"] < job.required_free_mib
                    ):
                        available.remove(gpu_info)
                        continue
                    pending.pop(pending_index); available.remove(gpu_info)
                    attempts[job.key] = attempts.get(job.key, 0) + 1
                    if job.log_path.exists() and job.log_path.stat().st_size:
                        job.log_path.replace(job.log_path.with_name(f"{job.log_path.stem}.attempt{attempts[job.key] - 1}.log"))
                    if job.kind in {"smoke", "patch"} and not is_done(job) and job.patch_primary.parent.exists(): shutil.rmtree(job.patch_primary.parent)
                    job.log_path.parent.mkdir(parents=True, exist_ok=True); log_file = job.log_path.open("w", encoding="utf-8")
                    process = subprocess.Popen(job.command, cwd=PROJECT_ROOT, env=environment(job, gpu_info["index"], config), stdout=log_file, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    gpu_reservations.reserve(gpu_info["index"], process.pid, "v3_full", job.key)
                finally:
                    fcntl.flock(launch_lock, fcntl.LOCK_UN)
                    launch_lock.close()
                running[job.key] = (process, job, log_file, gpu_info["index"])
                metadata.write(json.dumps({"time": time.strftime("%F %T"), "gpu": gpu_info["index"], "key": job.key, "command": job.command, "log": str(job.log_path)}, ensure_ascii=True) + "\n"); metadata.flush()
                print(f"[start] gpu={gpu_info['index']} {job.key}", flush=True)
            atomic_state({"status": "running", "time": time.strftime("%F %T"), "phase": phase_name, "completed": sum(is_done(job) for job in planned), "total": len(planned), "running": [{"key": key, "gpu": value[3], "pid": value[0].pid} for key, value in running.items()]})
            time.sleep(POLL_SECONDS)


def aggregate() -> int:
    rows = []
    for job in eval_jobs():
        parsed = parse_eval(job.log_path); episodes, successes = parsed if parsed else (0, 0); rate = successes / episodes if episodes else None
        rows.append({"table": job.table, "source_model": job.source_model, "target_model": job.target_model, "source_suite": job.source_suite, "target_suite": job.suite, "seed": job.seed, "episodes": episodes, "successes": successes, "success_rate": rate, "failure_rate": 1 - rate if rate is not None else None, "status": "complete" if episodes >= EXPECTED_EPISODES else "incomplete", "patch_primary": str(job.patch_primary), "patch_wrist": str(job.patch_wrist or job.patch_primary), "log_path": str(job.log_path)})
    fields = list(rows[0]); RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULTS_ROOT / "runs.csv").open("w", newline="", encoding="utf-8") as file: writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for table in ("table2", "table3", "table4", "table6"):
        selected = [row for row in rows if row["table"] == table]
        grouped = defaultdict(list)
        for row in selected:
            if row["status"] == "complete": grouped[(row["source_model"], row["target_model"], row["source_suite"], row["target_suite"])].append(row["failure_rate"])
        summary = []
        for key in sorted({(row["source_model"], row["target_model"], row["source_suite"], row["target_suite"]) for row in selected}):
            values = grouped.get(key, []); summary.append({"table": table, "source_model": key[0], "target_model": key[1], "source_suite": key[2], "target_suite": key[3], "num_seeds": len(values), "mean_failure_rate": mean(values) if values else None, "std_failure_rate": stdev(values) if len(values)>1 else (0.0 if values else None), "status": "complete" if len(values)==3 else "incomplete"})
        with (RESULTS_ROOT / f"{table}.csv").open("w", newline="", encoding="utf-8") as file: writer=csv.DictWriter(file, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        with (RESULTS_ROOT / f"{table}.md").open("w", encoding="utf-8") as file:
            file.write("| Source model | Target model | Source suite | Target suite | Seeds | Mean FR | Std FR | Status |\n|---|---|---|---|---:|---:|---:|---|\n")
            for row in summary:
                mean_fr="-" if row["mean_failure_rate"] is None else f"{row['mean_failure_rate']*100:.2f}%"; std_fr="-" if row["std_failure_rate"] is None else f"{row['std_failure_rate']*100:.2f}%"
                file.write(f"| {row['source_model']} | {row['target_model']} | {row['source_suite']} | {row['target_suite']} | {row['num_seeds']} | {mean_fr} | {std_fr} | {row['status']} |\n")
    metadata={"experiment":"scheme_one_v3_full","method":"TC-TD V3","models":list(MODELS),"suites":list(SUITES),"seeds":list(SEEDS),"patch_steps":PATCH_STEPS,"batch_size":BATCH_SIZE,"num_views":NUM_VIEWS,"perturbation_ratio":RATIO,"alpha":ALPHA,"step_size":STEP_SIZE,"relation_temperature":RELATION_TEMPERATURE,"episodes_per_task":EPISODES_PER_TASK,"expected_episodes_per_evaluation":EXPECTED_EPISODES,"physical_patch_jobs":len(patch_jobs()),"evaluation_jobs":len(eval_jobs()),"completed_evaluations":sum(row["status"]=="complete" for row in rows),"pi0_adaptation":"single SigLIP branch; cross-branch term omitted"}
    (RESULTS_ROOT / "metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=True)+"\n",encoding="utf-8"); print(f"Aggregated {metadata['completed_evaluations']}/{metadata['evaluation_jobs']}",flush=True); return 0


def main() -> int:
    args=parse_args()
    if args.mode=="aggregate": return aggregate()
    if args.mode=="full": return run_full()
    print(json.dumps({"smoke_jobs":len(smoke_jobs()),"patch_jobs":len(patch_jobs()),"evaluation_jobs":len(eval_jobs()),"table_counts":{table:sum(job.table==table for job in eval_jobs()) for table in ("table2","table3","table4","table6")}},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
