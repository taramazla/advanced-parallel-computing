#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Config (bisa di-edit)
# ==========================

SRC_FILE="conjugateGradient.cu"
EXE_FILE="conjugateGradient"

# Arsitektur GPU:
#   GTX 1080 Ti  -> sm_61
#   RTX 3080     -> sm_86
CUDA_ARCH="${CUDA_ARCH:-sm_61}"

# Default parameter CG (bisa di-override via argumen)
# Usage:
#   ./run_conjugateGradient.sh [NMin] [NMax] [NMult] [MAX_ITER] [EPS] [TOL]
N_MIN=${1:-256}
N_MAX=${2:-2048}
N_MULT=${3:-2}
MAX_ITER=${4:-1000}
EPS=${5:-1e-6}
TOL=${6:-1e-3}

echo "========================================"
echo " Conjugate Gradient CUDA Runner"
echo "========================================"
echo "Source      : $SRC_FILE"
echo "Binary      : $EXE_FILE"
echo "CUDA arch   : $CUDA_ARCH"
echo "Params      :"
echo "  NMin      = $N_MIN"
echo "  NMax      = $N_MAX"
echo "  NMult     = $N_MULT"
echo "  MAX_ITER  = $MAX_ITER"
echo "  EPS       = $EPS"
echo "  TOL       = $TOL"
echo "========================================"

# ==========================
# Compile
# ==========================
echo "[1/2] Compiling with nvcc..."
nvcc -O3 -arch="$CUDA_ARCH" -o "$EXE_FILE" "$SRC_FILE"

echo "[OK] Compile selesai."

# ==========================
# Run
# ==========================
echo "[2/2] Running CG..."
echo "./$EXE_FILE $N_MIN $N_MAX $N_MULT $MAX_ITER $EPS $TOL"
echo

"./$EXE_FILE" "$N_MIN" "$N_MAX" "$N_MULT" "$MAX_ITER" "$EPS" "$TOL"

echo
echo "Selesai. Cek file CSV log (conjugateGradient-*.csv) untuk hasil lengkap."
