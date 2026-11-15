#!/bin/sh
# Screenshots of the states: ./shoot.sh OUTDIR state... (all the states if none;
# server_login, server_account, server_admin, server_activity are the pages of tools/serve_repl).
# desktop in dark and light at 1440x1000, and mobile (390 px frame, coarse pointer).
# build.mjs must have been run into /tmp/repl_states.
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT=${1:-/tmp/repl_shots}; shift
STATES=${*:-$(sed -n '/const STATES = {/,/^  };/s/^    \([a-z_]*\): {.*/\1/p' "$(dirname "$0")/mock.js")}
mkdir -p "$OUT"
for s in $STATES; do
  case $s in server_*) page="${s}.html"; wrap="page=$s";; *) page="index.html?state=$s"; wrap="state=$s";; esac
  for t in dark:0 light:1; do
    name=${t%%:*}; n=${t##*:}
    "$CHROME" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
      --blink-settings=preferredColorScheme=$n --window-size=1440,1300 --virtual-time-budget=8000 \
      --screenshot="$OUT/$s-$name.png" "file:///tmp/repl_states/$page" 2>/dev/null
    "$CHROME" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
      --blink-settings=preferredColorScheme=$n,primaryPointerType=2,availablePointerTypes=2 --window-size=500,1700 --virtual-time-budget=8000 \
      --screenshot="$OUT/$s-$name-mobile.png" "file:///tmp/repl_states/wrapper.html?$wrap" 2>/dev/null
  done
done
