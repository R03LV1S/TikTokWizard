"""Universal Hardware Detection & Cross-Vendor Compute Profiles (v5.4).

Detects OS, CPU, RAM, physical GPUs (NVIDIA, AMD, Intel Arc, Apple Silicon),
installed PyTorch / ONNX Runtime / CTranslate2 accelerators, and usable FFmpeg
hardware encoders. Builds deterministic CPU or GPU Compute Profiles that wire
all pipeline stages (stem separation, MMS_FA, Echora, Whisper, FFmpeg render).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CORE_DIR = Path(__file__).resolve().parent
APP_ROOT = CORE_DIR.parent.parent  # DO NOT TOUCH
DEFAULT_PROFILE_PATH = APP_ROOT / "compute_profile.json"

STEM_MODEL_HQ = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
STEM_MODEL_FAST = "Kim_Vocal_2.onnx"

_VIRTUAL_GPU_KEYWORDS = (
    "virtual",
    "basic display",
    "basic render",
    "parsec",
    "splashtop",
    "citrix",
    "vmware",
    "vbox",
    "remote desktop",
    "idd",
    "spacedesk",
)

# Ordered vendor encoder preference per GPU vendor & OS
_VENDOR_ENCODERS: Dict[str, List[str]] = {
    "nvidia": ["h264_nvenc"],
    "amd": ["h264_amf", "h264_vaapi"],
    "intel": ["h264_qsv", "h264_vaapi"],
    "apple": ["h264_videotoolbox"],
}


@dataclass
class GpuDevice:
    """Represents one detected display/compute adapter."""

    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "other"
    name: str
    vram_gb: float = 0.0
    driver: str = ""
    is_virtual: bool = False


@dataclass
class HardwareReport:
    """Full snapshot of host hardware, runtimes, and working FFmpeg encoders."""

    os_name: str  # "windows" | "macos" | "linux"
    os_label: str
    arch: str
    hostname: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpus: List[GpuDevice] = field(default_factory=list)
    torch_version: str = "not installed"
    torch_cuda: bool = False
    torch_mps: bool = False
    torch_xpu: bool = False
    torch_rocm: bool = False
    torch_directml: bool = False
    ort_providers: List[str] = field(default_factory=list)
    ctranslate2_cuda: bool = False
    ffmpeg_available: bool = False
    ffmpeg_encoders: List[str] = field(default_factory=list)
    working_hw_encoders: List[str] = field(default_factory=list)

    def primary_gpu(self) -> Optional[GpuDevice]:
        physical = [g for g in self.gpus if not g.is_virtual and g.vendor in ("nvidia", "amd", "intel", "apple")]
        if not physical:
            return None
        priority = {"nvidia": 0, "apple": 1, "amd": 2, "intel": 3}
        physical.sort(key=lambda g: (priority.get(g.vendor, 9), -g.vram_gb))
        return physical[0]

    def has_capable_gpu(self) -> bool:
        return self.primary_gpu() is not None or self.torch_cuda or self.torch_mps or self.torch_xpu


@dataclass
class ComputeProfile:
    """Active execution profile selected by the user at startup or via CLI."""

    profile_id: str  # "cpu_only" | "nvidia_cuda" | "amd_directml" | "amd_rocm" | "intel_arc_dml" | "apple_mps"
    compute_mode: str  # "cpu" | "gpu"
    display_name: str
    gpu_vendor: str
    gpu_name: str
    cpu_name: str
    os_label: str
    stem_mode: str  # "hq" | "fast"
    stem_model: str
    stem_device: str  # "cuda" | "mps" | "xpu" | "dml" | "coreml" | "cpu"
    align_device: str  # "cuda" | "mps" | "xpu" | "cpu"
    whisper_device: str  # "cuda" | "cpu"
    whisper_compute_type: str  # "float16" | "int8"
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    video_codec: str = "libx264"
    bs_roformer_on_gpu: bool = False
    package_gaps: List[str] = field(default_factory=list)


def classify_gpu_vendor(name: str, compat: str = "") -> str:
    """Normalize raw GPU vendor strings across WMI, lspci, and macOS."""
    blob = f"{name} {compat}".lower()
    if any(k in blob for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
        return "nvidia"
    if any(k in blob for k in ("amd", "radeon", "ati ", "advanced micro devices")):
        return "amd"
    if "intel" in blob or "arc(" in blob or "arc " in blob or "uhd graphics" in blob or "iris" in blob:
        return "intel"
    if "apple" in blob and any(k in blob for k in ("m1", "m2", "m3", "m4", "m5", "gpu")):
        return "apple"
    return "other"


def _is_virtual_adapter(name: str) -> bool:
    low = (name or "").lower()
    return any(k in low for k in _VIRTUAL_GPU_KEYWORDS)


def _detect_cpu_ram() -> tuple[str, int, int, float]:
    threads = os.cpu_count() or 4
    cores = max(1, threads // 2) if threads >= 4 else threads
    cpu_name = platform.processor() or platform.machine() or "CPU"
    ram_gb = 0.0

    sys_name = platform.system().lower()
    try:
        if sys_name == "windows":
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "$c=Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors; "
                "$m=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
                "[PSCustomObject]@{Name=$c.Name;Cores=$c.NumberOfCores;Threads=$c.NumberOfLogicalProcessors;Ram=$m} | ConvertTo-Json -Compress",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            if r.returncode == 0 and r.stdout.strip():
                d = json.loads(r.stdout.strip())
                cpu_name = str(d.get("Name") or cpu_name).strip()
                cores = int(d.get("Cores") or cores)
                threads = int(d.get("Threads") or threads)
                ram_gb = round(float(d.get("Ram") or 0) / (1024**3), 1)
        elif sys_name == "darwin":
            r1 = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3)
            if r1.returncode == 0 and r1.stdout.strip():
                cpu_name = r1.stdout.strip()
            r2 = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=3)
            if r2.returncode == 0 and r2.stdout.strip().isdigit():
                cores = int(r2.stdout.strip())
            r3 = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
            if r3.returncode == 0 and r3.stdout.strip().isdigit():
                ram_gb = round(int(r3.stdout.strip()) / (1024**3), 1)
        else:
            if os.path.isfile("/proc/cpuinfo"):
                for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.lower().startswith("model name") and ":" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        break
            if os.path.isfile("/proc/meminfo"):
                for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024**2), 1)
                        break
    except Exception:
        pass
    return cpu_name, cores, threads, ram_gb


def _detect_gpus(sys_name: str, arch: str) -> List[GpuDevice]:
    gpus: List[GpuDevice] = []
    try:
        if sys_name == "windows":
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "$q=@{}; Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | ForEach-Object { if($_.DriverDesc -and $_.'HardwareInformation.qwMemorySize'){ $q[$_.DriverDesc]=[double]$_.'HardwareInformation.qwMemorySize' } }; "
                "Get-CimInstance Win32_VideoController | ForEach-Object { $v=$_.AdapterRAM; if($q.ContainsKey($_.Name) -and $q[$_.Name] -gt 0){ $v=$q[$_.Name] }; [PSCustomObject]@{Name=$_.Name;AdapterCompatibility=$_.AdapterCompatibility;DriverVersion=$_.DriverVersion;AdapterRAM=$v} } | ConvertTo-Json -Compress",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            if r.returncode == 0 and r.stdout.strip():
                raw = json.loads(r.stdout.strip())
                items = raw if isinstance(raw, list) else [raw]
                for it in items:
                    name = str(it.get("Name") or "").strip()
                    if not name:
                        continue
                    compat = str(it.get("AdapterCompatibility") or "")
                    drv = str(it.get("DriverVersion") or "")
                    vram_bytes = float(it.get("AdapterRAM") or 0)
                    vram_gb = round(vram_bytes / (1024**3), 1) if vram_bytes > 0 else 0.0
                    gpus.append(
                        GpuDevice(
                            vendor=classify_gpu_vendor(name, compat),
                            name=name,
                            vram_gb=vram_gb,
                            driver=drv,
                            is_virtual=_is_virtual_adapter(name),
                        )
                    )
        elif sys_name == "darwin":
            if "arm" in arch.lower():
                gpus.append(GpuDevice(vendor="apple", name="Apple Silicon Unified GPU"))
            else:
                r = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    data = json.loads(r.stdout)
                    for item in data.get("SPDisplaysDataType", []):
                        name = str(item.get("sppci_model") or "Mac GPU")
                        gpus.append(GpuDevice(vendor=classify_gpu_vendor(name), name=name))
        else:
            if shutil.which("nvidia-smi"):
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                if r.returncode == 0:
                    for line in r.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if parts and parts[0]:
                            vram = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 and parts[1].replace(".", "").isdigit() else 0.0
                            drv = parts[2] if len(parts) > 2 else ""
                            gpus.append(GpuDevice(vendor="nvidia", name=parts[0], vram_gb=vram, driver=drv))
            if not gpus and shutil.which("lspci"):
                r = subprocess.run(["lspci"], capture_output=True, text=True, timeout=4)
                if r.returncode == 0:
                    for line in r.stdout.splitlines():
                        if any(tag in line for tag in ("VGA compatible controller", "3D controller", "Display controller")):
                            name = line.split(":", 2)[-1].strip()
                            gpus.append(GpuDevice(vendor=classify_gpu_vendor(name), name=name))
    except Exception:
        pass
    return gpus


def _probe_ffmpeg_encoders(ffmpeg_bin: str = "ffmpeg", verify_hw: bool = True) -> tuple[bool, List[str], List[str]]:
    exe = ffmpeg_bin or "ffmpeg"
    if not shutil.which(exe) and not os.path.isfile(exe):
        return False, [], []
    listed: List[str] = []
    working: List[str] = []
    candidates = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "h264_vaapi", "libx264")
    try:
        r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
        out = r.stdout or ""
        for enc in candidates:
            if enc in out:
                listed.append(enc)
    except Exception:
        return True, [], []

    if "libx264" in listed:
        working.append("libx264")
    if verify_hw:
        for enc in listed:
            if enc == "libx264":
                continue
            try:
                cmd = [
                    exe,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=128x128:d=0.1:r=10",
                    "-pix_fmt",
                    "nv12" if enc in ("h264_qsv", "h264_vaapi") else "yuv420p",
                    "-c:v",
                    enc,
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ]
                pr = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
                if pr.returncode == 0:
                    working.append(enc)
            except Exception:
                pass
    else:
        working = list(listed)
    return True, listed, working


def detect_hardware(ffmpeg_bin: str = "ffmpeg", verify_encoders: bool = True) -> HardwareReport:
    """Inspect host OS, CPU, RAM, GPUs, ML backends, and usable FFmpeg encoders."""
    raw_sys = platform.system().lower()
    os_name = "windows" if raw_sys == "windows" else ("macos" if raw_sys == "darwin" else "linux")
    os_label = f"{platform.system()} {platform.release()} ({platform.machine()})"
    arch = platform.machine()
    hostname = socket.gethostname()

    cpu_name, cores, threads, ram_gb = _detect_cpu_ram()
    gpus = _detect_gpus(os_name, arch)

    torch_ver = "not installed"
    t_cuda = t_mps = t_xpu = t_rocm = t_dml = False
    try:
        import torch

        torch_ver = str(torch.__version__)
        t_cuda = bool(torch.cuda.is_available())
        t_rocm = bool(t_cuda and getattr(torch.version, "hip", None))
        t_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        t_xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        try:
            import torch_directml

            t_dml = bool(torch_directml.is_available())
        except Exception:
            t_dml = False
    except Exception:
        pass

    ort_providers: List[str] = []
    try:
        import onnxruntime as ort

        ort_providers = list(ort.get_available_providers())
    except Exception:
        pass

    ct2_cuda = False
    try:
        import ctranslate2

        ct2_cuda = bool(ctranslate2.get_cuda_device_count() > 0)
    except Exception:
        pass

    ff_ok, listed_enc, working_enc = _probe_ffmpeg_encoders(ffmpeg_bin, verify_hw=verify_encoders)
    return HardwareReport(
        os_name=os_name,
        os_label=os_label,
        arch=arch,
        hostname=hostname,
        cpu_name=cpu_name,
        cpu_cores=cores,
        cpu_threads=threads,
        ram_gb=ram_gb,
        gpus=gpus,
        torch_version=torch_ver,
        torch_cuda=t_cuda,
        torch_mps=t_mps,
        torch_xpu=t_xpu,
        torch_rocm=t_rocm,
        torch_directml=t_dml,
        ort_providers=ort_providers,
        ctranslate2_cuda=ct2_cuda,
        ffmpeg_available=ff_ok,
        ffmpeg_encoders=listed_enc,
        working_hw_encoders=working_enc,
    )


def _pick_encoder(vendor: str, report: HardwareReport) -> str:
    pool = report.working_hw_encoders or report.ffmpeg_encoders
    for cand in _VENDOR_ENCODERS.get(vendor, []):
        if cand in pool:
            return cand
    return "libx264"


def build_compute_profile(
    report: HardwareReport,
    mode: str = "gpu",
    stem_preference: str = "auto",
) -> ComputeProfile:
    """Resolve a concrete CPU or vendor GPU ComputeProfile from HardwareReport.

    mode: "cpu" | "gpu" | "auto"
    stem_preference: "auto" | "hq" | "fast"
    """
    req_mode = (mode or "gpu").strip().lower()
    pref = (stem_preference or "auto").strip().lower()
    primary = report.primary_gpu()

    if req_mode == "cpu" or (req_mode == "auto" and primary is None):
        s_mode = "fast" if pref == "fast" else "hq"
        return ComputeProfile(
            profile_id="cpu_only",
            compute_mode="cpu",
            display_name="CPU-Only Mode (Universal Safe, Studio BS-Roformer)",
            gpu_vendor="none",
            gpu_name=primary.name if primary else "CPU-Only",
            cpu_name=report.cpu_name,
            os_label=report.os_label,
            stem_mode=s_mode,
            stem_model=STEM_MODEL_FAST if s_mode == "fast" else STEM_MODEL_HQ,
            stem_device="cpu",
            align_device="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            onnx_providers=["CPUExecutionProvider"],
            video_codec="libx264",
            bs_roformer_on_gpu=False,
            package_gaps=[],
        )

    vendor = primary.vendor if primary else (
        "apple" if report.torch_mps else ("nvidia" if report.torch_cuda else "other")
    )
    gpu_name = primary.name if primary else "Detected GPU"
    gaps: List[str] = []

    if vendor == "nvidia" or (report.torch_cuda and not report.torch_rocm):
        bs_gpu = bool(report.torch_cuda)
        if not report.torch_cuda:
            gaps.append("Install CUDA PyTorch + onnxruntime-gpu to unlock ~20s BS-Roformer & MMS_FA on NVIDIA GPU")
        elif "CUDAExecutionProvider" not in report.ort_providers:
            gaps.append("Install onnxruntime-gpu for CUDA ONNX fast stems")
        s_mode = "fast" if pref == "fast" else "hq"
        s_dev = ("cuda" if s_mode == "hq" and bs_gpu else ("cuda" if "CUDAExecutionProvider" in report.ort_providers else "cpu"))
        return ComputeProfile(
            profile_id="nvidia_cuda",
            compute_mode="gpu",
            display_name=f"NVIDIA CUDA GPU Mode ({gpu_name})",
            gpu_vendor="nvidia",
            gpu_name=gpu_name,
            cpu_name=report.cpu_name,
            os_label=report.os_label,
            stem_mode=s_mode,
            stem_model=STEM_MODEL_FAST if s_mode == "fast" else STEM_MODEL_HQ,
            stem_device=s_dev,
            align_device="cuda" if report.torch_cuda else "cpu",
            whisper_device="cuda" if report.ctranslate2_cuda else "cpu",
            whisper_compute_type="float16" if report.ctranslate2_cuda else "int8",
            onnx_providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in report.ort_providers
                else ["CPUExecutionProvider"]
            ),
            video_codec=_pick_encoder("nvidia", report),
            bs_roformer_on_gpu=bs_gpu,
            package_gaps=gaps,
        )

    if vendor == "apple" or report.torch_mps:
        bs_gpu = bool(report.torch_mps)
        s_mode = "fast" if pref == "fast" else "hq"
        has_coreml = "CoreMLExecutionProvider" in report.ort_providers
        return ComputeProfile(
            profile_id="apple_mps",
            compute_mode="gpu",
            display_name=f"Apple Silicon MPS Mode ({gpu_name})",
            gpu_vendor="apple",
            gpu_name=gpu_name,
            cpu_name=report.cpu_name,
            os_label=report.os_label,
            stem_mode=s_mode,
            stem_model=STEM_MODEL_FAST if s_mode == "fast" else STEM_MODEL_HQ,
            stem_device="mps" if s_mode == "hq" and bs_gpu else ("coreml" if has_coreml else "cpu"),
            align_device="mps" if bs_gpu else "cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            onnx_providers=(
                ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                if has_coreml
                else ["CPUExecutionProvider"]
            ),
            video_codec=_pick_encoder("apple", report),
            bs_roformer_on_gpu=bs_gpu,
            package_gaps=gaps,
        )

    if vendor == "amd" and report.os_name == "linux":
        bs_gpu = bool(report.torch_cuda)
        if not bs_gpu:
            gaps.append("Install PyTorch ROCm build to run BS-Roformer & MMS_FA on AMD Linux GPU")
        s_mode = "fast" if pref == "fast" else "hq"
        return ComputeProfile(
            profile_id="amd_rocm",
            compute_mode="gpu",
            display_name=f"AMD ROCm GPU Mode ({gpu_name})",
            gpu_vendor="amd",
            gpu_name=gpu_name,
            cpu_name=report.cpu_name,
            os_label=report.os_label,
            stem_mode=s_mode,
            stem_model=STEM_MODEL_FAST if s_mode == "fast" else STEM_MODEL_HQ,
            stem_device="cuda" if s_mode == "hq" and bs_gpu else "cpu",
            align_device="cuda" if bs_gpu else "cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            onnx_providers=["CPUExecutionProvider"],
            video_codec=_pick_encoder("amd", report),
            bs_roformer_on_gpu=bs_gpu,
            package_gaps=gaps,
        )

    if vendor in ("amd", "intel"):
        has_dml = "DmlExecutionProvider" in report.ort_providers
        if report.os_name == "windows" and not has_dml:
            gaps.append("Install onnxruntime-directml to unlock ~25s Kim_Vocal_2 GPU stems on Windows AMD/Intel")
        # In GPU mode on Windows DirectML: default to fast GPU stems unless "hq" explicitly requested
        s_mode = "hq" if pref == "hq" else ("fast" if has_dml else "hq")
        pid = "intel_arc_dml" if vendor == "intel" else "amd_directml"
        align_dev = "xpu" if (vendor == "intel" and report.torch_xpu) else "cpu"
        return ComputeProfile(
            profile_id=pid,
            compute_mode="gpu",
            display_name=(
                f"{'Intel Arc' if vendor == 'intel' else 'AMD Radeon'} GPU Mode ({gpu_name})"
            ),
            gpu_vendor=vendor,
            gpu_name=gpu_name,
            cpu_name=report.cpu_name,
            os_label=report.os_label,
            stem_mode=s_mode,
            stem_model=STEM_MODEL_FAST if s_mode == "fast" else STEM_MODEL_HQ,
            stem_device=("dml" if s_mode == "fast" and has_dml else "cpu"),
            align_device=align_dev,
            whisper_device="cpu",
            whisper_compute_type="int8",
            onnx_providers=(
                ["DmlExecutionProvider", "CPUExecutionProvider"]
                if has_dml
                else ["CPUExecutionProvider"]
            ),
            video_codec=_pick_encoder(vendor, report),
            bs_roformer_on_gpu=False,
            package_gaps=gaps,
        )

    # Unknown GPU -> safe CPU fallback
    return build_compute_profile(report, mode="cpu", stem_preference=pref)


def profile_from_dict(data: Dict[str, Any]) -> ComputeProfile:
    valid_keys = {f.name for f in ComputeProfile.__dataclass_fields__.values()}
    filtered = {k: v for k, v in (data or {}).items() if k in valid_keys}
    return ComputeProfile(**filtered)


def save_compute_profile(
    profile: ComputeProfile, path: Optional[Path] = None
) -> Path:
    target = Path(path or DEFAULT_PROFILE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_compute_profile(path: Optional[Path] = None) -> Optional[ComputeProfile]:
    target = Path(path or DEFAULT_PROFILE_PATH)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("profile_id"):
            return profile_from_dict(data)
    except Exception:
        pass
    return None


def resolve_active_profile(
    ffmpeg_bin: str = "ffmpeg",
    verify_encoders: bool = False,
) -> ComputeProfile:
    """Resolve runtime ComputeProfile honoring TW_COMPUTE_MODE env override or saved disk profile."""
    env_mode = os.environ.get("TW_COMPUTE_MODE", "").strip().lower()
    env_stem = os.environ.get("TW_STEM_MODEL", "").strip()
    pref = "fast" if env_stem == STEM_MODEL_FAST else ("hq" if env_stem == STEM_MODEL_HQ else "auto")

    if env_mode in ("cpu", "gpu"):
        rep = detect_hardware(ffmpeg_bin=ffmpeg_bin, verify_encoders=verify_encoders)
        return build_compute_profile(rep, mode=env_mode, stem_preference=pref)

    saved = load_compute_profile()
    if saved is not None:
        if pref in ("hq", "fast") and pref != saved.stem_mode:
            saved.stem_mode = pref
            saved.stem_model = STEM_MODEL_FAST if pref == "fast" else STEM_MODEL_HQ
        return saved

    rep = detect_hardware(ffmpeg_bin=ffmpeg_bin, verify_encoders=verify_encoders)
    mode = "gpu" if rep.has_capable_gpu() else "cpu"
    return build_compute_profile(rep, mode=mode, stem_preference=pref or "auto")


def get_torch_device(preferred: Optional[str] = None) -> str:
    """Resolve active PyTorch device ('cuda', 'mps', 'xpu', or 'cpu') safely."""
    dev = (preferred or "").strip().lower()
    if not dev:
        try:
            dev = resolve_active_profile(verify_encoders=False).align_device
        except Exception:
            dev = "cpu"
    try:
        import torch

        if dev == "cuda" and torch.cuda.is_available():
            return "cuda"
        if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if dev == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def install_gpu_packages_for_vendor(
    vendor: str,
    os_name: str,
    python_exe: Optional[str] = None,
    log_fn: Callable[[str], None] = print,
) -> bool:
    """Install vendor-appropriate GPU packages into the active venv (opt-in helper)."""
    py = python_exe or sys.executable
    cmds: List[List[str]] = []
    if vendor == "nvidia":
        cmds.append([
            py, "-m", "pip", "install", "--upgrade",
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124",
        ])
        cmds.append([py, "-m", "pip", "install", "--upgrade", "onnxruntime-gpu"])
    elif vendor in ("amd", "intel") and os_name == "windows":
        # Replace CPU onnxruntime cleanly with onnxruntime-directml for DirectML GPU acceleration
        cmds.append([py, "-m", "pip", "uninstall", "-y", "onnxruntime"])
        cmds.append([py, "-m", "pip", "install", "--upgrade", "onnxruntime-directml"])
    elif vendor == "amd" and os_name == "linux":
        cmds.append([
            py, "-m", "pip", "install", "--upgrade",
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/rocm6.2",
        ])
    else:
        log_fn("[Hardware] Current platform already uses standard PyTorch / ONNX packages.")
        return True

    for cmd in cmds:
        log_fn(f"[Hardware] Running: {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log_fn(f"[Hardware] Command failed (exit {r.returncode}): {' '.join(cmd)}")
            return False
    log_fn("[Hardware] GPU runtime packages installed successfully.")
    return True
