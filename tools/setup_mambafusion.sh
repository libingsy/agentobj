#!/usr/bin/env bash
# Build the MambaFusion environment (Python 3.8, torch 2.1, CUDA 11.8).
#
# MambaFusion needs its own environment: its CUDA kernels are built against
# torch 2.1 / CUDA 11.8, while the Agent-OBJ environment uses torch 2.5.
# Agent-OBJ talks to it through scripts/mambafusion_worker.py.
#
# Requirements: conda, nvcc 11.8 on PATH (or CUDA_HOME set), gcc <= 11.
#
# Usage:
#   bash tools/setup_mambafusion.sh                 # env name "mambafusion"
#   ENV_NAME=mf CUDA_ARCHS=89 bash tools/setup_mambafusion.sh
#
# CUDA_ARCHS restricts the kernels to your GPU(s), e.g. 86 (A6000/3090),
# 89 (L40/4090), 90 (H100). Comma separated. Empty builds the upstream default list.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${ROOT}/third_party/MambaFusion"
ENV_NAME="${ENV_NAME:-mambafusion}"
CUDA_ARCHS="${CUDA_ARCHS:-}"

if [[ ! -f "${REPO}/setup.py" ]]; then
  echo "MambaFusion is missing. Run: git submodule update --init --recursive"
  exit 1
fi

# Let the custom kernels target the GPUs listed in CUDA_ARCHS.
cd "${REPO}"
if git apply --check "${ROOT}/third_party/patches/mambafusion_cuda_archs.patch" 2>/dev/null; then
  git apply "${ROOT}/third_party/patches/mambafusion_cuda_archs.patch"
  echo "[setup] applied mambafusion_cuda_archs.patch"
fi
if [[ -n "${CUDA_ARCHS}" ]]; then
  export AGENTOBJ_CUDA_ARCHS="${CUDA_ARCHS}"
  export TORCH_CUDA_ARCH_LIST="$(echo "${CUDA_ARCHS}" | sed -E 's/([0-9])([0-9])/\1.\2/g; s/,/;/g')"
fi
export MAX_JOBS="${MAX_JOBS:-4}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.8
fi
conda activate "${ENV_NAME}"

python -m pip install -U pip 'setuptools<70' wheel ninja packaging
if ! python -c "import torch; assert torch.__version__.startswith('2.1.0')" >/dev/null 2>&1; then
  python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
fi
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

python -m pip install \
  'numpy==1.23.5' 'llvmlite==0.39.1' 'numba==0.56.4' \
  easydict pyyaml tqdm 'opencv-python==4.8.1.78' pyquaternion pillow \
  'shapely==1.8.4' 'kornia==0.7.3' scikit-image tensorboardX SharedArray \
  'timm==1.0.8' 'transformers==4.43.3' fvcore yacs termcolor einops \
  spconv-cu118 'nuscenes-devkit==1.0.5' 'av2==0.2.1' ipdb \
  'https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp38-cp38-linux_x86_64.whl'

cd "${REPO}"
python -m pip install -e . --no-deps --no-build-isolation
python mambafusion_setup.py build_ext --inplace
cd "${REPO}/selective_scan"
python -m pip install -e . --no-deps --no-build-isolation
cd "${REPO}/mamba_diffv/mamba"
python -m pip install -e . --no-deps --no-build-isolation

cd "${ROOT}"
echo "[setup] done. Set paths.mambafusion_python in configs/full.yaml to:"
echo "        $(command -v python)"
echo "[setup] After placing the checkpoint, check the build with: python scripts/smoke_mambafusion.py"
