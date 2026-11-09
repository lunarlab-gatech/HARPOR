#!/bin/bash
# Open the HARPOR web app of a Slurm cluster on your computer (macOS or Linux). Export these once in your shell profile:
#
#   HARPOR_SSH_HOST     user@login-node of the cluster (required)
#   HARPOR_REMOTE_CODE  the repo on the login node (required). Quote a value with a ~, such as '~/code/HARPOR', so
#                       that the login node expands it
#   HARPOR_SSH_SOCKET   the ControlMaster socket (default ~/.ssh/cm-harpor)
#
# Log in to the cluster first, so that the ControlMaster socket is up (ssh -M -S SOCKET HOST), then:
#
#   bash harpor/webapp/open_slurm.sh JOBID [LOCAL_PORT]      (LOCAL_PORT defaults to 8765)
#
# It runs start_slurm.sh on the login node over that socket (start or reuse the server in job JOBID),
# forwards localhost:LOCAL_PORT to the server's node and port on the same socket and opens
# http://localhost:LOCAL_PORT/?token=... in the browser, which sets the login cookie. The forward
# lasts as long as the ControlMaster. The last forward of each local port is kept in
# ~/.config/harpor/webapp_forward_<port>, so a later run cancels it when the node or port changed.
set -euo pipefail
JOB=${1:?usage: open_slurm.sh JOBID [LOCAL_PORT]}
LPORT=${2:-8765}
[[ $JOB =~ ^[0-9]+$ && $LPORT =~ ^[0-9]+$ ]] || { echo "JOBID and LOCAL_PORT must be numbers" >&2; exit 2; }
SOCK=${HARPOR_SSH_SOCKET:-~/.ssh/cm-harpor}
HOST=${HARPOR_SSH_HOST:?set HARPOR_SSH_HOST=user@login-node}
REMOTE=${HARPOR_REMOTE_CODE:?set HARPOR_REMOTE_CODE to the repo on the login node}  # the remote shell expands a ~
STATE=~/.config/harpor/webapp_forward_$LPORT

if ! ssh -S "$SOCK" -O check "$HOST" 2>/dev/null; then
  echo "no ControlMaster on $SOCK; log in to the cluster first (ssh -M -S $SOCK $HOST)" >&2
  exit 1
fi
echo "ControlMaster $SOCK is up; starting the server in job $JOB"
out=$(ssh -S "$SOCK" "$HOST" "bash $REMOTE/harpor/webapp/start_slurm.sh $JOB")
NODE=$(sed -n 's/^NODE=//p' <<< "$out")
PORT=$(sed -n 's/^PORT=//p' <<< "$out")
TOKEN=$(sed -n 's/^TOKEN=//p' <<< "$out")
if [ -z "$NODE" ] || [ -z "$PORT" ] || [ -z "$TOKEN" ]; then
  echo "start_slurm.sh gave no NODE, PORT and TOKEN" >&2
  exit 1
fi

NEW=$LPORT:$NODE:$PORT
OLD=$(cat "$STATE" 2>/dev/null || true)
if [ -n "$OLD" ] && [ "$OLD" != "$NEW" ]; then
  if ssh -S "$SOCK" -O cancel -L "$OLD" "$HOST" 2>/dev/null; then
    echo "cancelled the older forward localhost:${OLD/:/ -> }"
  fi
fi
if ! ssh -S "$SOCK" -O forward -L "$NEW" "$HOST"; then
  echo "the forward failed; is localhost:$LPORT taken? Check with: lsof -nP -iTCP:$LPORT -sTCP:LISTEN" >&2
  exit 1
fi
mkdir -p -m 700 ~/.config/harpor
echo "$NEW" > "$STATE"
echo "forwarding localhost:$LPORT -> $NODE:$PORT"
URL="http://localhost:$LPORT/?token=$TOKEN"
if command -v xdg-open >/dev/null; then xdg-open "$URL"; else open "$URL"; fi
echo "opened http://localhost:$LPORT/ in the browser"
