#!/usr/bin/env bash
# Run after conda activation and before a costly InstantID generation job.
set -euo pipefail

echo "=== NVIDIA SMI Status ==="
nvidia-smi

echo "=== Python Environment & Hardware Verification ==="
python - <<'PY'
import onnxruntime as ort
import torch
import diffusers

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"Diffusers: {diffusers.__version__}")
print(f"ONNX Providers: {ort.get_available_providers()}")

assert torch.cuda.is_available(), "CRITICAL: PyTorch cannot see the allocated GPU"
assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "CRITICAL: Install a CUDA-compatible onnxruntime-gpu in this environment"
)

device = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Active GPU: {device}")
print(f"Total VRAM: {vram_gb:.2f} GB")

# Brief tensor allocation test to ensure CUDA bridge isn't corrupted
try:
    _ = torch.zeros((1, 1), device="cuda")
    print("CUDA Allocation Test: Passed")
except Exception as e:
    raise RuntimeError(f"CRITICAL: GPU is visible but tensor allocation failed: {e}")
PY