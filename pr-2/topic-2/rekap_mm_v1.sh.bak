#!/bin/sh
# rekap_mm.sh
# Rekap hasil eksperimen matmul dari log run_mm_experiments.sh
# Output: tabel waktu eksekusi & speedup (format Markdown)

LOG_FILE="${1:-hasil_eksperimen_matmul.txt}"

if [ ! -f "$LOG_FILE" ]; then
  echo "Usage: $0 hasil_eksperimen_matmul.txt"
  echo "File '$LOG_FILE' tidak ditemukan."
  exit 1
fi

awk '
function fmt(x) {
  if (x == "" || x != x || x == 0) return "NaN";
  return sprintf("%.3f", x);
}
function spd(ts, tp) {
  if (ts == "" || ts != ts || ts == 0) return "NaN";
  if (tp == "" || tp != tp || tp == 0) return "NaN";
  return sprintf("%.2f×", ts/tp);
}

BEGIN {
  currN = 0;
  section = "";
  mpi_np = 0;
  tilesize = 0;
  blocksize = 0;
}

/^N = [0-9]+/ {
  currN = $3 + 0;
  section = "";
  mpi_np = 0;
  tilesize = 0;
  blocksize = 0;
  next;
}

/^\[SEQ\]/ {
  section = "seq";
  next;
}

/^\[CUDA-NOPT\]/ {
  section = "nopt";
  blocksize = 0;
  # Parse blockSize
  if ($0 ~ /blockSize=[0-9]+/) {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^blockSize=/) {
        split($i, a, "=");
        blocksize = a[2] + 0;
        break;
      }
    }
  }
  next;
}

/^\[CUDA-OPT-SHARED\]/ {
  section = "shared";
  tilesize = 0;
  # Parse TILE_SIZE
  if ($0 ~ /TILE_SIZE=[0-9]+/) {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^TILE_SIZE=/) {
        split($i, a, "=");
        tilesize = a[2] + 0;
        break;
      }
    }
  }
  next;
}

/^\[CUBLAS\]/ {
  section = "cublas";
  next;
}

/^\[MPI\]/ {
  section = "mpi";
  mpi_np = 0;
  # Parse: [MPI] N=256, np=8
  if ($0 ~ /np=[0-9]+/) {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^np=/) {
        split($i, a, "=");
        mpi_np = a[2] + 0;
        break;
      }
    }
  }
  next;
}

section == "seq" && /CPU compute time:/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      seq[currN] = $i + 0.0;
      if (!(currN in seen)) {
        seen[currN] = 1;
        Ns[++nN] = currN;
      }
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Total Execution Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      t = $i + 0.0;
      if (section == "nopt") {
        key = currN "_" blocksize;
        if (!(key in nopt) || t < nopt[key]) {
          nopt[key] = t;
          if (!(blocksize in nopt_tiles_seen)) {
            nopt_tiles_seen[blocksize] = 1;
            nopt_tiles[++n_nopt_tiles] = blocksize;
          }
        }
      } else if (section == "shared") {
        key = currN "_" tilesize;
        if (!(key in shared) || t < shared[key]) {
          shared[key] = t;
          if (!(tilesize in shared_tiles_seen)) {
            shared_tiles_seen[tilesize] = 1;
            shared_tiles[++n_shared_tiles] = tilesize;
          }
        }
      } else if (section == "cublas") {
        cublas[currN] = t;
      } else if (section == "mpi") {
        key = currN "_" mpi_np;
        mpi[key] = t;
        if (!(mpi_np in mpi_nps_seen)) {
          mpi_nps_seen[mpi_np] = 1;
          mpi_nps[++n_mpi_nps] = mpi_np;
        }
      }
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Computation Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      tcomp = $i + 0.0;
      if (section == "nopt") {
        key = currN "_" blocksize;
        nopt_comp[key] = tcomp;
      } else if (section == "shared") {
        key = currN "_" tilesize;
        shared_comp[key] = tcomp;
      } else if (section == "cublas") {
        cublas_comp[currN] = tcomp;
      } else if (section == "mpi") {
        key = currN "_" mpi_np;
        mpi_comp[key] = tcomp;
      }
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Communication Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      tcomm = $i + 0.0;
      if (section == "nopt") {
        key = currN "_" blocksize;
        nopt_comm[key] = tcomm;
      } else if (section == "shared") {
        key = currN "_" tilesize;
        shared_comm[key] = tcomm;
      } else if (section == "cublas") {
        cublas_comm[currN] = tcomm;
      } else if (section == "mpi") {
        key = currN "_" mpi_np;
        mpi_comm[key] = tcomm;
      }
      break;
    }
  }
  next;
}

END {
  # Sort N secara ascending (bubble sort sederhana, N kecil saja)
  for (i = 1; i <= nN; i++) {
    for (j = i + 1; j <= nN; j++) {
      if (Ns[j] < Ns[i]) {
        tmp = Ns[i];
        Ns[i] = Ns[j];
        Ns[j] = tmp;
      }
    }
  }
  
  # Sort tile sizes
  for (i = 1; i <= n_nopt_tiles; i++) {
    for (j = i + 1; j <= n_nopt_tiles; j++) {
      if (nopt_tiles[j] < nopt_tiles[i]) {
        tmp = nopt_tiles[i];
        nopt_tiles[i] = nopt_tiles[j];
        nopt_tiles[j] = tmp;
      }
    }
  }
  
  for (i = 1; i <= n_shared_tiles; i++) {
    for (j = i + 1; j <= n_shared_tiles; j++) {
      if (shared_tiles[j] < shared_tiles[i]) {
        tmp = shared_tiles[i];
        shared_tiles[i] = shared_tiles[j];
        shared_tiles[j] = tmp;
      }
    }
  }
  
  # Sort MPI process counts
  for (i = 1; i <= n_mpi_nps; i++) {
    for (j = i + 1; j <= n_mpi_nps; j++) {
      if (mpi_nps[j] < mpi_nps[i]) {
        tmp = mpi_nps[i];
        mpi_nps[i] = mpi_nps[j];
        mpi_nps[j] = tmp;
      }
    }
  }

  # ============================
  # Tabel waktu eksekusi CUDA no-shared (berbagai blockSize)
  # ============================
  print "## Tabel Waktu Eksekusi CUDA no-shared (ms) - Berbagai blockSize";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf(" blockSize=%2d |", nopt_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_nopt_tiles; t++) {
      key = Nval "_" nopt_tiles[t];
      printf(" %11s |", fmt(nopt[key]));
    }
    print "";
  }
  print "";

  # ============================
  # Tabel waktu eksekusi CUDA shared (berbagai TILE_SIZE)
  # ============================
  print "## Tabel Waktu Eksekusi CUDA shared (ms) - Berbagai TILE_SIZE";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf(" TILE_SIZE=%2d |", shared_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_shared_tiles; t++) {
      key = Nval "_" shared_tiles[t];
      printf(" %11s |", fmt(shared[key]));
    }
    print "";
  }
  print "";

  # ============================
  # Tabel waktu eksekusi MPI (berbagai np)
  # ============================
  print "## Tabel Waktu Eksekusi MPI (ms) - Berbagai np";
  print "";
  printf("| N    |");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf(" np=%2d |", mpi_nps[p]);
  }
  print "";
  printf("|------|");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf("------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      printf(" %4s |", fmt(mpi[key]));
    }
    print "";
  }
  print "";

  # ============================
  # Tabel waktu eksekusi ringkasan (best dari setiap kategori)
  # ============================
  print "## Tabel Waktu Eksekusi Ringkasan (ms) - Best dari setiap kategori";
  print "";
  print "| N    | CPU Seq | CUDA no-shared | CUDA shared | cuBLAS | MPI (best) |";
  print "|------|--------:|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    tseq   = seq[Nval];
    
    # Find best nopt
    best_nopt = "";
    for (t = 1; t <= n_nopt_tiles; t++) {
      key = Nval "_" nopt_tiles[t];
      if (key in nopt) {
        if (best_nopt == "" || nopt[key] < best_nopt) {
          best_nopt = nopt[key];
        }
      }
    }
    
    # Find best shared
    best_shared = "";
    for (t = 1; t <= n_shared_tiles; t++) {
      key = Nval "_" shared_tiles[t];
      if (key in shared) {
        if (best_shared == "" || shared[key] < best_shared) {
          best_shared = shared[key];
        }
      }
    }
    
    tcublas = cublas[Nval];
    
    # Find best MPI
    best_mpi = "";
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      if (key in mpi) {
        if (best_mpi == "" || mpi[key] < best_mpi) {
          best_mpi = mpi[key];
        }
      }
    }

    printf("| %-4d | %7s | %13s | %10s | %7s | %9s |\n",
           Nval,
           fmt(tseq),
           fmt(best_nopt),
           fmt(best_shared),
           fmt(tcublas),
           fmt(best_mpi));
  }

  print "";
  # ============================
  # Tabel speedup ringkasan
  # ============================
  print "## Tabel Speedup terhadap CPU sekuensial (Ts/Tp) - Best dari setiap kategori";
  print "";
  print "| N    | CUDA no-shared | CUDA shared | cuBLAS | MPI (best) |";
  print "|------|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    tseq   = seq[Nval];
    
    # Find best nopt
    best_nopt = "";
    for (t = 1; t <= n_nopt_tiles; t++) {
      key = Nval "_" nopt_tiles[t];
      if (key in nopt) {
        if (best_nopt == "" || nopt[key] < best_nopt) {
          best_nopt = nopt[key];
        }
      }
    }
    
    # Find best shared
    best_shared = "";
    for (t = 1; t <= n_shared_tiles; t++) {
      key = Nval "_" shared_tiles[t];
      if (key in shared) {
        if (best_shared == "" || shared[key] < best_shared) {
          best_shared = shared[key];
        }
      }
    }
    
    tcublas = cublas[Nval];
    
    # Find best MPI
    best_mpi = "";
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      if (key in mpi) {
        if (best_mpi == "" || mpi[key] < best_mpi) {
          best_mpi = mpi[key];
        }
      }
    }

    printf("| %-4d | %13s | %10s | %7s | %9s |\n",
           Nval,
           spd(tseq, best_nopt),
           spd(tseq, best_shared),
           spd(tseq, tcublas),
           spd(tseq, best_mpi));
  }

  print "";
  # ============================
  # Tabel Speedup MPI (berbagai np)
  # ============================
  print "## Tabel Speedup MPI terhadap CPU sekuensial - Berbagai np";
  print "";
  printf("| N    |");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf(" np=%2d |", mpi_nps[p]);
  }
  print "";
  printf("|------|");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf("------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    tseq = seq[Nval];
    printf("| %-4d |", Nval);
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      printf(" %4s |", spd(tseq, mpi[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Computation Time - CUDA no-shared
  # ============================
  print "## Tabel Computation Time CUDA no-shared (ms)";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf(" blockSize=%2d |", nopt_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_nopt_tiles; t++) {
      key = Nval "_" nopt_tiles[t];
      printf(" %11s |", fmt(nopt_comp[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Communication Time - CUDA no-shared
  # ============================
  print "## Tabel Communication Time CUDA no-shared (ms)";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf(" blockSize=%2d |", nopt_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_nopt_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_nopt_tiles; t++) {
      key = Nval "_" nopt_tiles[t];
      printf(" %11s |", fmt(nopt_comm[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Computation Time - CUDA shared
  # ============================
  print "## Tabel Computation Time CUDA shared (ms)";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf(" TILE_SIZE=%2d |", shared_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_shared_tiles; t++) {
      key = Nval "_" shared_tiles[t];
      printf(" %11s |", fmt(shared_comp[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Communication Time - CUDA shared
  # ============================
  print "## Tabel Communication Time CUDA shared (ms)";
  print "";
  printf("| N    |");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf(" TILE_SIZE=%2d |", shared_tiles[t]);
  }
  print "";
  printf("|------|");
  for (t = 1; t <= n_shared_tiles; t++) {
    printf("-------------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (t = 1; t <= n_shared_tiles; t++) {
      key = Nval "_" shared_tiles[t];
      printf(" %11s |", fmt(shared_comm[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Computation Time - MPI
  # ============================
  print "## Tabel Computation Time MPI (ms)";
  print "";
  printf("| N    |");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf(" np=%2d |", mpi_nps[p]);
  }
  print "";
  printf("|------|");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf("------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      printf(" %4s |", fmt(mpi_comp[key]));
    }
    print "";
  }

  print "";
  # ============================
  # Tabel Communication Time - MPI
  # ============================
  print "## Tabel Communication Time MPI (ms)";
  print "";
  printf("| N    |");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf(" np=%2d |", mpi_nps[p]);
  }
  print "";
  printf("|------|");
  for (p = 1; p <= n_mpi_nps; p++) {
    printf("------:|");
  }
  print "";

  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    printf("| %-4d |", Nval);
    for (p = 1; p <= n_mpi_nps; p++) {
      key = Nval "_" mpi_nps[p];
      printf(" %4s |", fmt(mpi_comm[key]));
    }
    print "";
  }
}
' "$LOG_FILE"
