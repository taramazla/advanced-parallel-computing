#!/bin/sh
# rekap_mm.sh
# Rekap hasil eksperimen matmul dari log run_mm_experiments.sh
# Output: tabel waktu eksekusi per N dengan semua variant (format tabel)

LOG_FILE="${1:-hasil_eksperimen_matmul.txt}"

if [ ! -f "$LOG_FILE" ]; then
  echo "Usage: $0 hasil_eksperimen_matmul.txt"
  echo "File '$LOG_FILE' tidak ditemukan."
  exit 1
fi

awk '
function fmt(x) {
  if (x == "" || x != x) return "-";
  return sprintf("%.3f", x);
}
function spd(ts, tp) {
  if (ts == "" || ts != ts || ts == 0) return "-";
  if (tp == "" || tp != tp || tp == 0) return "-";
  return sprintf("%.2fx", ts/tp);
}

BEGIN {
  currN = 0;
  section = "";
  variant = "";
  rowIdx = 0;
}

/^#+$/ { next; }

/^N = [0-9]+/ || /^########/ && /[0-9]+/ {
  # Extract N value
  if (match($0, /[0-9]+/)) {
    currN = substr($0, RSTART, RLENGTH) + 0;
    if (!(currN in seenN)) {
      seenN[currN] = 1;
      Ns[++nN] = currN;
    }
  }
  section = "";
  variant = "";
  next;
}

/^\[SEQ\]/ {
  section = "seq";
  variant = "-";
  next;
}

/^\[CUDA-NOPT\]/ {
  section = "nopt";
  # Extract blockSize
  if (match($0, /blockSize=[0-9]+/)) {
    variant = "block=" substr($0, RSTART+10, RLENGTH-10);
  }
  next;
}

/^\[CUDA-OPT-SHARED\]/ {
  section = "shared";
  # Extract TILE_SIZE
  if (match($0, /TILE_SIZE=[0-9]+/)) {
    variant = "tile=" substr($0, RSTART+10, RLENGTH-10);
  }
  next;
}

/^\[CUBLAS\]/ {
  section = "cublas";
  variant = "-";
  next;
}

/^\[MPI\]/ {
  section = "mpi";
  # Extract np
  if (match($0, /np=[0-9]+/)) {
    np = substr($0, RSTART+3, RLENGTH-3);
    variant = "np=" np;
  }
  next;
}

section == "seq" && /CPU compute time:/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      t = $i + 0.0;
      rowIdx++;
      row_N[rowIdx] = currN;
      row_method[rowIdx] = "CPU Seq";
      row_variant[rowIdx] = "-";
      row_total[rowIdx] = t;
      row_comp[rowIdx] = t;
      row_comm[rowIdx] = 0.0;
      seq_time[currN] = t;
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Total Execution Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      pending_total = $i + 0.0;
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Computation Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      pending_comp = $i + 0.0;
      break;
    }
  }
  next;
}

(section == "nopt" || section == "shared" || section == "cublas" || section == "mpi") && /Communication Time/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^[0-9]+\.?[0-9]*$/) {
      pending_comm = $i + 0.0;
      break;
    }
  }
  # Now we have all three values, store the row
  rowIdx++;
  row_N[rowIdx] = currN;
  if (section == "nopt") {
    row_method[rowIdx] = "CUDA NOPT";
  } else if (section == "shared") {
    row_method[rowIdx] = "CUDA Shared";
  } else if (section == "cublas") {
    row_method[rowIdx] = "cuBLAS";
  } else if (section == "mpi") {
    row_method[rowIdx] = "MPI";
  }
  row_variant[rowIdx] = variant;
  row_total[rowIdx] = pending_total;
  row_comp[rowIdx] = pending_comp;
  row_comm[rowIdx] = pending_comm;
  next;
}

END {
  # Sort N ascending
  for (i = 1; i <= nN; i++) {
    for (j = i + 1; j <= nN; j++) {
      if (Ns[j] < Ns[i]) {
        tmp = Ns[i];
        Ns[i] = Ns[j];
        Ns[j] = tmp;
      }
    }
  }

  # Print tables per N
  for (k = 1; k <= nN; k++) {
    Nval = Ns[k];
    tseq = seq_time[Nval];

    print "";
    print "N = " Nval;
    print "";
    printf("%-14s %-14s %12s %12s %12s %14s\n", "Method", "Variant", "Total(ms)", "Comp(ms)", "Comm(ms)", "Speedup_vs_CPU");
    print "";

    # Print rows for this N
    for (r = 1; r <= rowIdx; r++) {
      if (row_N[r] == Nval) {
        speedup = spd(tseq, row_total[r]);
        printf("%-14s %-14s %12s %12s %12s %14s\n",
               row_method[r],
               row_variant[r],
               fmt(row_total[r]),
               fmt(row_comp[r]),
               fmt(row_comm[r]),
               speedup);
      }
    }
    print "";
  }
}
' "$LOG_FILE"
