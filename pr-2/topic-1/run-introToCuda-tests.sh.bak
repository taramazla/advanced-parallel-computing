#!/usr/bin/env bash

# Script untuk menguji introToCuda.cu dengan variasi GRID_SIZE (blocks) dan BLOCK_SIZE (threads)
# Asumsi: executable bernama introToCuda (hasil kompilasi nvcc introToCuda.cu -o introToCuda)

EXE=./introToCuda

if [ ! -x "$EXE" ]; then
  echo "Executable $EXE tidak ditemukan atau tidak bisa dieksekusi."
  echo "Kompilasi dulu dengan:"
  echo "  nvcc introToCuda.cu -o introToCuda"
  exit 1
fi

# Daftar kombinasi (BLOCKS, THREADS)
cases=(
  "1 5"
  "3 5"
  "5 3"
  "2 8"
  "4 4"
  "3 10"
)

for c in "${cases[@]}"; do
  b=$(echo "$c" | awk '{print $1}')
  t=$(echo "$c" | awk '{print $2}')
  echo "============================================================"
  echo "GRID_SIZE (blocks)  = $b"
  echo "BLOCK_SIZE (threads)= $t"
  echo "Total elemen (size) = $((b * t))"
  echo "------------------------------------------------------------"
  $EXE "$t" "$b"
  echo
done