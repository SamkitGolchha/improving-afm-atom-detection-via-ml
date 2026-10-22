#!/usr/bin/env bash
# Download Track B inputs (approved 2026-10-16): ice pretrained weights + AFM-ice simulated training sets.
# Resumable; each file MD5-verified; one fresh re-download if a checksum fails. Deletes nothing else.
set -u
RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
W="$RAW/ice_weights_zenodo10054348"; S="$RAW/afm_ice_sims"
mkdir -p "$W" "$S"

get() {  # dir name md5 url
  local f="$1/$2" want="$3" url="$4" got
  for attempt in 1 2; do
    for i in $(seq 1 30); do
      curl -sS -L -C - --retry 5 --retry-delay 10 -o "$f" "$url" && break
      sleep 15
    done
    got=$(md5 -q "$f")
    if [ "$got" = "$want" ]; then echo "OK   $2 $(stat -f %z "$f") B md5 $got"; return 0; fi
    echo "BAD  $2 md5 $got (want $want), attempt $attempt"
    [ "$attempt" = 1 ] && mv "$f" "$f.bad_md5"   # keep the bad copy (delete nothing), re-download fresh
  done
  echo "FAIL $2 after re-download"; return 1
}

Z=https://zenodo.org/api/records
get "$W" weights_ice-au111-monolayer.pth f358ee74320ad4fae5cd924fbce7c9d2 "$Z/10054348/files/weights_ice-au111-monolayer.pth/content"
get "$W" weights_ice-au111-bilayer.pth   affaee9622887962c4e6987c1a265265 "$Z/10054348/files/weights_ice-au111-bilayer.pth/content"
get "$W" weights_ice-cu111.pth           457dd8610bb11f8992a32746192a0fb9 "$Z/10054348/files/weights_ice-cu111.pth/content"
get "$S" AFM-ice-Au111-monolayer.tar.gz  67626324403348227a00ddff82ff3963 "$Z/10049832/files/AFM-ice-Au111-monolayer.tar.gz/content"
get "$S" AFM-ice-Au111-bilayer.tar.gz    2abdaaff5fe31e94386293dde31675dc "$Z/10049856/files/AFM-ice-Au111-bilayer.tar.gz/content"
get "$S" AFM-ice-Cu111.tar.gz            bde7ad1ea08f49843df76dcc327fc895 "$Z/10047850/files/AFM-ice-Cu111.tar.gz/content"
echo DONE
