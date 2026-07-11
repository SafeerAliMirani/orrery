#!/bin/bash
# Orrery Phase 1: pull ~1M Gaia DR3 6D stars from the AIP mirror.
# Run on the Galileo100 LOGIN node (compute nodes have no internet).
# Chunked async TAP jobs over random_index windows; resumable: finished
# chunk files are skipped on rerun. Produces gaia_1m.csv.
#
#   bash pull_gaia.sh /g100_work/EIRI_E_UNISA2/smirani0/orrery/data

set -u
OUT=${1:?usage: pull_gaia.sh OUTDIR}
mkdir -p "$OUT/chunks"
TAP="https://gaia.aip.de/tap/async"
COLS="source_id,ra,dec,parallax,parallax_over_error,pmra,pmdec,radial_velocity,radial_velocity_error,ruwe,phot_g_mean_mag,bp_rp"
CUTS="radial_velocity IS NOT NULL AND parallax_over_error>=10 AND ruwe<1.4 AND radial_velocity_error<5"
STEP=10000000          # 10M random_index per chunk, ~92k rows each
NCHUNK=11              # 0..110M total, expect ~1.016M rows

for ((i=0; i<NCHUNK; i++)); do
  f="$OUT/chunks/chunk_$(printf '%02d' $i).csv"
  if [ -s "$f" ]; then echo "chunk $i exists, skip"; continue; fi
  lo=$((i * STEP)); hi=$((lo + STEP - 1))
  q="SELECT $COLS FROM gaiadr3.gaia_source WHERE $CUTS AND random_index BETWEEN $lo AND $hi"
  echo "chunk $i: random_index $lo..$hi"
  job=$(curl -s -i -X POST "$TAP" \
    --data-urlencode "REQUEST=doQuery" --data-urlencode "LANG=ADQL" \
    --data-urlencode "FORMAT=csv" --data-urlencode "PHASE=RUN" \
    --data-urlencode "QUERY=$q" \
    | grep -i '^Location:' | tr -d '\r' | awk '{print $2}')
  if [ -z "$job" ]; then echo "  submit failed"; exit 1; fi
  for ((p=0; p<120; p++)); do
    sleep 5
    phase=$(curl -s "$job/phase")
    [ "$phase" = "COMPLETED" ] && break
    if [ "$phase" = "ERROR" ] || [ "$phase" = "ABORTED" ]; then
      echo "  job $phase"; curl -s "$job/error" | head -5; exit 1
    fi
  done
  [ "$phase" != "COMPLETED" ] && { echo "  timed out"; exit 1; }
  curl -s -L "$job/results/result" -o "$f"
  echo "  $(wc -l < "$f") lines"
done

# merge: header from chunk 0, bodies from all
head -1 "$OUT/chunks/chunk_00.csv" > "$OUT/gaia_1m.csv"
for f in "$OUT"/chunks/chunk_*.csv; do tail -n +2 "$f" >> "$OUT/gaia_1m.csv"; done
n=$(($(wc -l < "$OUT/gaia_1m.csv") - 1))
echo "gaia_1m.csv: $n stars"
