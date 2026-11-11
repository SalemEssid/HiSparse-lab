"""Record machine facts for the G1 report.

Captures: GPU model / compute capability / VRAM, PCIe link, virtualization mode,
driver and library versions, usable SDPA backends, host RAM, CPU, and how much
pinned host memory can actually be allocated (the full KV cache lives there).

Usage:
    python hw/env_report.py --out hw/results/env.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sh(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def host_memory_gb() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_gb": round(vm.total / 2**30, 1), "available_gb": round(vm.available / 2**30, 1)}
    except ImportError:
        info = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, val = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = round(int(val.split()[0]) / 2**20, 1)
        except OSError:
            pass
        return {"total_gb": info.get("MemTotal"), "available_gb": info.get("MemAvailable")}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def probe_sdpa_backends() -> dict:
    """Try each SDPA backend on a small causal bf16 problem shaped like Qwen3 attention."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return {"error": "torch.nn.attention.sdpa_kernel not available (torch < 2.3)"}

    backends = {
        "flash": getattr(SDPBackend, "FLASH_ATTENTION", None),
        "efficient": getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
        "math": getattr(SDPBackend, "MATH", None),
    }
    q = torch.randn(1, 32, 256, 128, device="cuda", dtype=torch.bfloat16)
    result = {}
    for name, be in backends.items():
        if be is None:
            result[name] = "not in this torch build"
            continue
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
            result[name] = "ok"
        except Exception as e:  # noqa: BLE001
            result[name] = "unavailable: " + str(e).splitlines()[0][:160]
    return result


def probe_pinned_gb(limit_gb: float) -> dict:
    """Allocate pinned host memory in 1 GiB chunks until failure or limit, then free it."""
    chunks, t0 = [], time.perf_counter()
    err = None
    try:
        while len(chunks) < int(limit_gb):
            chunks.append(torch.empty(2**30, dtype=torch.uint8, pin_memory=True))
    except Exception as e:  # noqa: BLE001
        err = str(e).splitlines()[0][:160]
    got = len(chunks)
    secs = time.perf_counter() - t0
    del chunks
    return {
        "requested_limit_gb": int(limit_gb),
        "allocated_gb": got,
        "alloc_seconds": round(secs, 2),
        "stopped_by": err or "limit reached",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw/results/env.json")
    ap.add_argument(
        "--pinned-probe-gb",
        type=float,
        default=32,
        help="upper limit for the pinned-memory probe; also capped at 60%% of available RAM",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available to torch"
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    mem = host_memory_gb()

    try:
        import transformers

        tf_version = transformers.__version__
    except ImportError:
        tf_version = None

    pinned_limit = args.pinned_probe_gb
    if mem.get("available_gb"):
        pinned_limit = min(pinned_limit, 0.6 * mem["available_gb"])

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": tf_version,
        "gpu": {
            "name": prop.name,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "vram_gb": round(prop.total_memory / 2**30, 2),
            "sm_count": prop.multi_processor_count,
            "device_count": torch.cuda.device_count(),
        },
        "nvidia_smi": sh(
            "nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,"
            "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv"
        ),
        "virtualization": sh("nvidia-smi -q | grep -i -A2 'virtualization'"),
        "gpu_topology": sh("nvidia-smi topo -m"),
        "host_memory": mem,
        "memlock_ulimit": sh("ulimit -l"),
        "cpu": {"model": cpu_model(), "logical_cores": os.cpu_count()},
        "lscpu_numa": sh("lscpu | grep -i -E 'numa|socket'"),
        "sdpa_backends": probe_sdpa_backends(),
        "pinned_probe": probe_pinned_gb(pinned_limit),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    g = report["gpu"]
    md = [
        "## Machine",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| GPU | {g['name']} (sm {g['compute_capability']}, {g['vram_gb']} GiB, {g['sm_count']} SMs, {g['device_count']} device(s)) |",
        f"| torch / CUDA / transformers | {report['torch']} / {report['torch_cuda']} / {report['transformers']} |",
        f"| Host RAM total / available | {mem.get('total_gb')} / {mem.get('available_gb')} GiB |",
        f"| CPU | {report['cpu']['model']} ({report['cpu']['logical_cores']} logical cores) |",
        f"| Pinned alloc probe | {report['pinned_probe']['allocated_gb']} GiB of {report['pinned_probe']['requested_limit_gb']} requested ({report['pinned_probe']['stopped_by']}) |",
        f"| memlock ulimit | {report['memlock_ulimit']} |",
        "| SDPA backends | " + ", ".join(f"{k}: {v}" for k, v in report["sdpa_backends"].items()) + " |",
        "",
        "```text",
        report["nvidia_smi"],
        report["virtualization"],
        "```",
        "",
    ]
    md_text = "\n".join(md)
    out.with_suffix(".md").write_text(md_text)
    print(md_text)


if __name__ == "__main__":
    main()
