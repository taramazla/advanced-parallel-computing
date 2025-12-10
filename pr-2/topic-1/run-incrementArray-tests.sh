#!/usr/bin/env bash

# Script uji incrementArray.cu untuk berbagai N dan blockSize
# Asumsi sudah ada executable bernama incrementArray
#   nvcc incrementArray.cu -o incrementArray

EXE=./incrementArray

if [ ! -x "$EXE" ]; then
  echo "Executable $EXE tidak ditemukan atau tidak bisa dieksekusi."
  echo "Kompilasi dulu dengan:"
  echo "  nvcc incrementArray.cu -o incrementArray"
  exit 1
fi

# Daftar N yang akan diuji (kecil → besar)
Ns=(
  1000000000
  10000000000
)

# Daftar blockSize yang akan diuji
blockSizes=(
  32
  64
  128
  256
  512
  1024
)

for N in "${Ns[@]}"; do
  for bs in "${blockSizes[@]}"; do
    echo "============================================================"
    echo "Uji: N = $N, blockSize = $bs"
    echo "------------------------------------------------------------"
    # Jalankan program; jika terjadi error (misalnya melebihi kapasitas GPU),
    # akan tampak dari pesan error / assert.
    $EXE "$N" "$bs"
    status=$?
    if [ $status -ne 0 ]; then
      echo ">> Program exit dengan status $status (mungkin error kapasitas / assert)."
    fi
    echo
  done
done