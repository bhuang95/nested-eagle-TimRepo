#!/bin/bash
# List date/times over a fixed window at two different intervals and diff them.
#
#   (a) 2016-02-01T06 -> 2017-01-31T18 every 54 h
#   (b) 2016-02-01T06 -> 2017-01-31T18 every 78 h
#   (c) times in (a) but not in (b)

set -u

START="2016-02-01T06"
END="2017-01-31T18"

FMT="%Y-%m-%dT%H"

# gen_times <interval_hours>  -> one time per line
gen_times() {
    local step=$1
    local t_epoch end_epoch
    t_epoch=$(date -u -d "${START/T/ }:00:00" +%s)
    end_epoch=$(date -u -d "${END/T/ }:00:00" +%s)
    while [ "$t_epoch" -le "$end_epoch" ]; do
        date -u -d "@$t_epoch" +"$FMT"
        t_epoch=$(( t_epoch + step * 3600 ))
    done
}

outdir=$(mktemp -d)
trap 'rm -rf "$outdir"' EXIT

gen_times 54 > "$outdir/a.txt"
gen_times 78 > "$outdir/b.txt"
comm -23 "$outdir/a.txt" "$outdir/b.txt" > "$outdir/a_not_b.txt"   # both lists are sorted

echo "=== (a) ${START} -> ${END}, every 54 h  [$(wc -l < "$outdir/a.txt") times] ==="
cat "$outdir/a.txt"
echo
echo "=== (b) ${START} -> ${END}, every 78 h  [$(wc -l < "$outdir/b.txt") times] ==="
cat "$outdir/b.txt"
echo
echo "=== (c) in (a) but NOT in (b)  [$(wc -l < "$outdir/a_not_b.txt") times] ==="
cat "$outdir/a_not_b.txt"
