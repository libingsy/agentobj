#!/usr/bin/env bash
# Download the public checkpoints Agent-OBJ uses.
#
#   pretrained/LGM/model_fp16_fixrot.safetensors   LGM reconstruction model
#   pretrained/yolov10x.pt                         YOLOv10x (monocular 2D detector)
#
# MV-Adapter and SDXL are fetched by diffusers from Hugging Face on first use.
# MambaFusion weights are on Google Drive and must be downloaded manually
# (see README, "Checkpoints").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/pretrained/LGM"

fetch() {
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "have ${dest}"
    return 0
  fi
  echo "download ${url}"
  wget -q --show-progress -O "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
}

fetch "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors" \
  "${ROOT}/pretrained/LGM/model_fp16_fixrot.safetensors"
fetch "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt" \
  "${ROOT}/pretrained/yolov10x.pt"

echo "done"
