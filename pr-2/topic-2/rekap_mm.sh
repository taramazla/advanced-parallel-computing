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
}

/^N = [0-9]+/ {
  currN = $3 + 0;
  section = "";
  mpi_np = 0;
  next;
}

/^\[SEQ\]/ {
  section = "seq";
  next;
}

/^\[CUDA-NOPT\]/ {
  section = "nopt";
  next;
}

/^\[CUDA-OPT-SHARED\]/ {
  section = "shared";
  next;
}

/^\[CUBLAS\]/ {
  section = "cublas";
  next;
}

/^\[MPI\]/ {
  section = "";
  mpi_np = 0;
  # Parse: [MPI] N=256, np=8
  if ($0 ~ /np=[0-9]+/) {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^np=/) {
        split($i, a, "=");
        mpi_np = a[2] + 0;
        if (mpi_np == 8) {
          section = "mpi8";
        }
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

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi8") && /Total Execution Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      t = $i + 0.0;
      if (section == "nopt") {
        if (!(currN in nopt) || t < nopt[currN]) {
          nopt[currN] = t;
        }
      } else if (section == "shared") {
        if (!(currN in shared) || t < shared[currN]) {
          shared[currN] = t;
        }
      } else if (section == "cublas") {
        cublas[currN] = t;
      } else if (section == "mpi8") {
        mpi8[currN] = t;
      }
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi8") && /Computation Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      tcomp = $i + 0.0;
      if (section == "nopt") {
        nopt_comp[currN] = tcomp;
      } else if (section == "shared") {
        shared_comp[currN] = tcomp;
      } else if (section == "cublas") {
        cublas_comp[currN] = tcomp;
      } else if (section == "mpi8") {
        mpi8_comp[currN] = tcomp;
      }
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi8") && /Communication Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      tcomm = $i + 0.0;
      if (section == "nopt") {
        nopt_comm[currN] = tcomm;
      } else if (section == "shared") {
        shared_comm[currN] = tcomm;
      } else if (section == "cublas") {
        cublas_comm[currN] = tcomm;
      } else if (section == "mpi8") {
        mpi8_comm[currN] = tcomm;
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

  # ============================
  # Tabel waktu eksekusi
  # ============================
  print "## Tabel Waktu Eksekusi (ms)";
  print "";
  print "| N    | CPU Seq | CUDA no-shared | CUDA shared | cuBLAS | MPI (np=8) |";
  print "|------|--------:|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    tseq   = seq[Nval];
    tnopt  = nopt[Nval];
    tsh    = shared[Nval];
    tcublas= cublas[Nval];
    tmpi8  = mpi8[Nval];

    printf("| %-4d | %7s | %13s | %10s | %7s | %9s |\n",
           Nval,
           fmt(tseq),
           fmt(tnopt),
           fmt(tsh),
           fmt(tcublas),
           fmt(tmpi8));
  }

  print "";
  # ============================
  # Tabel speedup
  # ============================
  print "## Tabel Speedup terhadap CPU sekuensial (Ts/Tp)";
  print "";
  print "| N    | CUDA no-shared | CUDA shared | cuBLAS | MPI (np=8) |";
  print "|------|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    tseq   = seq[Nval];
    tnopt  = nopt[Nval];
    tsh    = shared[Nval];
    tcublas= cublas[Nval];
    tmpi8  = mpi8[Nval];

    printf("| %-4d | %13s | %10s | %7s | %9s |\n",
           Nval,
           spd(tseq, tnopt),
           spd(tseq, tsh),
           spd(tseq, tcublas),
           spd(tseq, tmpi8));
  }

  print "";
  # ============================
  # Tabel Computation Time (ms)
  # ============================
  print "## Tabel Computation Time (ms)";
  print "";
  print "| N    | CUDA no-shared | CUDA shared | cuBLAS | MPI (np=8) |";
  print "|------|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    printf("| %-4d | %13s | %10s | %7s | %9s |\n",
           Nval,
           fmt(nopt_comp[Nval]),
           fmt(shared_comp[Nval]),
           fmt(cublas_comp[Nval]),
           fmt(mpi8_comp[Nval]));
  }

  print "";
  # ============================
  # Tabel Communication Time (ms)
  # ============================
  print "## Tabel Communication Time (ms)";
  print "";
  print "| N    | CUDA no-shared | CUDA shared | cuBLAS | MPI (np=8) |";
  print "|------|---------------:|------------:|-------:|-----------:|";

  for (k = 1; k <= nN; k++) {
    Nval   = Ns[k];
    printf("| %-4d | %13s | %10s | %7s | %9s |\n",
           Nval,
           fmt(nopt_comm[Nval]),
           fmt(shared_comm[Nval]),
           fmt(cublas_comm[Nval]),
           fmt(mpi8_comm[Nval]));
  }
}
' "$LOG_FILE"
