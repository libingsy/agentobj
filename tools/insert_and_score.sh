#!/usr/bin/env bash
# Generate one pedestrian, then insert it into several val keyframes and score
# YOLO and MambaFusion. insert_and_score.py starts the MambaFusion worker itself,
# so this script stays in the Agent-OBJ environment. Set paths.mambafusion_python
# in the config.
#
#   AGENTOBJ_PYTHON=~/miniconda3/envs/agentobj/bin/python bash tools/insert_and_score.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTOBJ_PYTHON="${AGENTOBJ_PYTHON:-python}"
CONFIG="${CONFIG:-configs/full.yaml}"
OUT="${OUT:-outputs/insert_object}"
SCENES="${SCENES:-4}"
KEYFRAMES="${KEYFRAMES:-1}"
SEMANTIC="${SEMANTIC:-carrying a box, holding a shopping bag}"
STYLE="${STYLE:-wearing a dark jacket and jeans}"

cd "${ROOT}"
"${AGENTOBJ_PYTHON}" scripts/generate_object.py \
  --config "${CONFIG}" --semantic "${SEMANTIC}" --style "${STYLE}" --output "${OUT}"
"${AGENTOBJ_PYTHON}" scripts/insert_and_score.py \
  --config "${CONFIG}" --object "${OUT}" --output "${OUT}/baseline" \
  --scenes "${SCENES}" --keyframes "${KEYFRAMES}"
