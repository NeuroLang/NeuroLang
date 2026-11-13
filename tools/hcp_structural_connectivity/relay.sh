#!/bin/bash
# Relay the HCP connectomes through this machine when the cluster's own
# downloads are slow (download.py): each file is downloaded here, copied to
# the staging directory of the cluster (as .part, renamed when whole), and
# deleted here; parcellate.py takes it from there. Subjects already
# parcellated (OUT/<subject>.npz) or staged are skipped, so the relay
# resumes. The Glasser matrices go to STAGING/glasser.
#
#   relay.sh HOST STAGING OUT LOCAL_DIR [JOBS]
#   relay.sh margaret /scratch/dwasserm/hcp_sc/staging \
#       /data/parietal/store3/work/dwasserm/hcp_sc/out /tmp/hcp_relay 4
set -uo pipefail
HOST=$1 STAGING=$2 OUT=$3 LOCAL=$4 JOBS=${5:-4}
BUCKET="https://swift.rc.nectar.org.au/v1/AUTH_ee5989f4c9184ea29012bb124cd3dff0/connectome_storage"
SSH=(ssh -F "$HOME/ssh_config_dw" -o ServerAliveInterval=30)
SCP=(scp -q -F "$HOME/ssh_config_dw" -o ServerAliveInterval=30)
CACERT=${CACERT:-/etc/ssl/cert.pem}
mkdir -p "$LOCAL"
export HOST STAGING LOCAL BUCKET CACERT
export SSH_CMD="${SSH[*]}" SCP_CMD="${SCP[*]}"

# the objects to relay, but those already done or staged on the cluster
listing() {
  local marker=""
  while :; do
    page=$(curl -sf --cacert "$CACERT" -m 120 \
      "$BUCKET?prefix=HCP_1200/&limit=10000&marker=$marker") || return 1
    [ -z "$page" ] && return 0
    printf '%s\n' "$page"
    marker=$(printf '%s\n' "$page" | tail -1)
  done
}
done_here=$("${SSH[@]}" "$HOST" "ls $OUT $STAGING $STAGING/glasser 2>/dev/null")
listing | grep -E '/(high_resolution|glasser_atlas)/structural_connectivity/' \
  | while read -r name; do
      file=${name##*/}
      if [[ $name == */high_resolution/* ]]; then
        subject=${name#HCP_1200/}; subject=${subject%%/*}
        grep -qx "$subject.npz" <<<"$done_here" && continue
        echo "$name $subject.npz"
      else
        grep -qx "$file" <<<"$done_here" && continue
        echo "$name glasser/$file"
      fi
    done > "$LOCAL/todo.txt"
echo "$(wc -l < "$LOCAL/todo.txt") files to relay"

relay_one() {
  local name=$1 target=$2 local_file
  local_file="$LOCAL/${target//\//_}"
  for attempt in 1 2 3 4 5; do
    if curl -sf --cacert "$CACERT" -m 3600 -o "$local_file" "$BUCKET/$name" \
        && $SCP_CMD "$local_file" "$HOST:$STAGING/$target.part" \
        && $SSH_CMD "$HOST" "mv $STAGING/$target.part $STAGING/$target"; then
      rm -f "$local_file"
      echo "$target"
      return 0
    fi
    echo "$target: failed (try $attempt)" >&2
    sleep $((30 * attempt))
  done
  rm -f "$local_file"
  return 1
}
export -f relay_one
# the connectomes first, the Glasser matrices after them
sort -k2,2 -t' ' "$LOCAL/todo.txt" | awk '{print ($2 ~ /^glasser/ ? 1 : 0), $0}' \
  | sort -s -k1,1 | cut -d' ' -f2- \
  | xargs -P "$JOBS" -L 1 bash -c 'relay_one "$0" "$1"'
"${SSH[@]}" "$HOST" "touch $STAGING/DONE"
echo "relayed"
