#!/bin/bash
# run_mm_experiments.sh
# Eksperimen perkalian matriks:
# - Seq (CPU)
# - CUDA tanpa shared memory (nopt)
# - CUDA dengan shared memory (opt_TILE)
# - cuBLAS
# - MPI (cluster / multicore)

set -e  # stop kalau ada error

#############################
# KONFIGURASI EKSPERIMEN   #
#############################

# Ukuran matriks yang diuji
N_VALUES=(256 512 1024 2048 4096 8192 16384 32768)

N_VALUES_SEQ_MAX=2048  # Maks N untuk seq (CPU) agar tidak terlalu lama

# Block size 1D untuk CUDA no-shared (nopt)
BLOCK_SIZES=(8 16 32)

# TILE_SIZE (block 2D) untuk CUDA shared (opt)
TILE_SIZES=(8 16 32)

# Jumlah proses untuk MPI
MPI_PROCS=(1 2 4 8)

# Kalau jalan sebagai root di container dan OpenMPI protes,
# set ini menjadi "yes" agar menambahkan --allow-run-as-root
ALLOW_RUN_AS_ROOT="yes"

LOG_FILE="hasil_eksperimen_matmul.txt"

#############################
# COMPILATION              #
#############################

echo "=== Kompilasi program ==="

# Sequential CPU
echo "Compile seq.cu ..."
nvcc seq.cu -O3 -o seq

# CUDA tanpa shared memory
echo "Compile nopt.cu ..."
nvcc nopt.cu -O3 -o nopt

# CUDA dengan shared memory: beberapa TILE_SIZE
for T in "${TILE_SIZES[@]}"; do
  echo "Compile opt.cu dengan TILE_SIZE=${T} ..."
  nvcc -O3 -DTILE_SIZE=${T} opt.cu -o opt_${T}
done

# cuBLAS
echo "Compile cublas.cu ..."
nvcc cublas.cu -O3 -lcublas -o cublas

# MPI
if command -v mpicc >/dev/null 2>&1; then
  echo "Compile mpi_mm.c (MPI) ..."
  mpicc mpi_mm.c -O3 -o mpi_mm
else
  echo "Peringatan: mpicc tidak ditemukan, bagian MPI akan dilewati."
fi

echo
echo "=== Mulai EKSPERIMEN ==="
echo "Log akan disimpan ke: ${LOG_FILE}"
echo "========================="

# Bersihkan log lama
echo "# Hasil eksperimen perkalian matriks" > "${LOG_FILE}"
echo "# Tanggal: $(date)" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

########################################
# LOOP EKSPERIMEN UNTUK SETIAP N      #
########################################

for N in "${N_VALUES[@]}"; do
  echo
  echo "########################################################" | tee -a "${LOG_FILE}"
  echo "N = ${N}" | tee -a "${LOG_FILE}"
  echo "########################################################" | tee -a "${LOG_FILE}"
  echo >> "${LOG_FILE}"

  ##########################
  # 1. SEQUENTIAL (CPU)   #
  ##########################
  if [ ${N} -le ${N_VALUES_SEQ_MAX} ]; then
    echo "[SEQ] CPU sequential, N=${N}" | tee -a "${LOG_FILE}"
    ./seq "${N}" | tee -a "${LOG_FILE}"
    echo | tee -a "${LOG_FILE}"
  else
    echo "[SEQ] CPU sequential, N=${N} - SKIPPED (N > ${N_VALUES_SEQ_MAX})" | tee -a "${LOG_FILE}"
    echo | tee -a "${LOG_FILE}"
  fi

  ########################################
  # 2. CUDA TANPA SHARED MEMORY (NOPt)  #
  ########################################
  for BS in "${BLOCK_SIZES[@]}"; do
    echo "[CUDA-NOPT] N=${N}, blockSize=${BS}" | tee -a "${LOG_FILE}"
    ./nopt "${N}" "${BS}" | tee -a "${LOG_FILE}"
    echo | tee -a "${LOG_FILE}"
  done

  ######################################
  # 3. CUDA DENGAN SHARED MEMORY      #
  ######################################
  for T in "${TILE_SIZES[@]}"; do
    EXE="opt_${T}"
    if [ -x "${EXE}" ]; then
      echo "[CUDA-OPT-SHARED] N=${N}, TILE_SIZE=${T}" | tee -a "${LOG_FILE}"
      ./"${EXE}" "${N}" | tee -a "${LOG_FILE}"
      echo | tee -a "${LOG_FILE}"
    fi
  done

  ##########################
  # 4. cuBLAS              #
  ##########################
  echo "[CUBLAS] N=${N}" | tee -a "${LOG_FILE}"
  ./cublas "${N}" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"

  ##########################
  # 5. MPI (CPU CLUSTER)   #
  ##########################
  if [ -x "./mpi_mm" ] && command -v mpirun >/dev/null 2>&1; then
    if [ ${N} -le ${N_VALUES_SEQ_MAX} ]; then
      for P in "${MPI_PROCS[@]}"; do
        echo "[MPI] N=${N}, np=${P}" | tee -a "${LOG_FILE}"

        if [ "${ALLOW_RUN_AS_ROOT}" = "yes" ]; then
          mpirun --allow-run-as-root -np "${P}" ./mpi_mm "${N}" | tee -a "${LOG_FILE}"
        else
          mpirun -np "${P}" ./mpi_mm "${N}" | tee -a "${LOG_FILE}"
        fi

        echo | tee -a "${LOG_FILE}"
      done
    else
      echo "[MPI] N=${N} - SKIPPED (N > ${N_VALUES_SEQ_MAX})" | tee -a "${LOG_FILE}"
      echo | tee -a "${LOG_FILE}"
    fi
  else
    echo "[MPI] Dilewati (mpi_mm atau mpirun tidak tersedia)" | tee -a "${LOG_FILE}"
  fi

done

echo
echo "=== SELESAI ==="
echo "Semua output tersimpan di: ${LOG_FILE}"
