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
  if (x == "" || x != x) return "NaN";
  return sprintf("%.3f", x);
}
function spd(ts, tp) {
  if (ts == "" || ts != ts || ts == 0) return "NaN";
  if (tp == "" || tp != tp || tp == 0) return "NaN";
  return sprintf("%.2f×", ts/tp);
}

BEGIN {
  INF = 1e30;
}

{
  line = $0;

  # Deteksi N = ...
  if (match(line, /^N[[:space:]]*=[[:space:]]*([0-9]+)/, m)) {
    currN = m[1] + 0;
    section = "";
    mpi_np = 0;
    next;
  }

  # Section markers
  if (line ~ /^\[SEQ\]/) {
    section = "seq";
    next;
  }

  if (line ~ /^\[CUDA-NOPT\]/) {
    section = "nopt";
    next;
  }

  if (line ~ /^\[CUDA-OPT-SHARED\]/) {
    section = "shared";
    next;
  }

  if (line ~ /^\[CUBLAS\]/) {
    section = "cublas";
    next;
  }

  if (line ~ /^\[MPI\]/) {
    # Contoh: [MPI] N=256, np=8
    section = "";
    mpi_np = 0;
    if (match(line, /\[MPI\][[:space:]]*N=([0-9]+),[[:space:]]*np=([0-9]+)/, mmpi)) {
      currN = mmpi[1] + 0;
      mpi_np = mmpi[2] + 0;
      if (mpi_np == 8) {
        section = "mpi8";
      }
    }
    next;
  }

  # Ambil waktu CPU sekuensial
  if (section == "seq") {
    if (match(line, /CPU compute time:[[:space:]]*([0-9.]+)[[:space:]]*ms/, mt)) {
      t = mt[1] + 0.0;
      seq[currN] = t;
      if (!(currN in seen)) {
        seen[currN] = 1;
        Ns[++nN] = currN;
      }
    }
    next;
  }

  # Ambil waktu CUDA & cuBLAS (in milliseconds)
  if (section == "nopt" || section == "shared" || section == "cublas" || section == "mpi8") {
    if (match(line, /Total Execution Time[[:space:]]*=[[:space:]]*([0-9.]+)[[:space:]]*ms/, mt2)) {
      t = mt2[1] + 0.0;
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
    }
    next;
  }
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
}
' "$LOG_FILE"
