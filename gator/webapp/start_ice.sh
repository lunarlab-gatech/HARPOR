#!/bin/bash
# Start the HARPOR web app inside a running Slurm job on PACE ICE. Run it on the login node:
#
#   bash ~/scratch/gator/gator_clone/gator/webapp/start_ice.sh JOBID [PORT]
#
# The server runs on the job's compute node as a job step (srun --overlap), so the pipeline stages it
# starts share the job's GPU. It writes ~/.config/gator/webapp.json (node, port, token, job); home is
# shared between the login and compute nodes, so this script reads it here. PORT 0 (the default) lets
# the server pick a free port. A server from an earlier start is reused only when webapp.json says it
# runs in job JOBID on one of its nodes (slurm_job, slurm_node) and it still answers; a webapp.json
# from another job (the server deletes it when it stops, but not when it is killed) is ignored.
# Messages go to stderr; stdout ends with NODE=, PORT= and TOKEN= lines for open_mac.sh.
#
# One-time setup of envs/web, the server's Python:
#   G=$(readlink -f ~/scratch)/gator
#   $G/tools/micromamba create -y -p $G/envs/web -c conda-forge python=3.12 pip
#   $G/envs/web/bin/pip install fastapi==0.143.2 uvicorn==0.54.0 av==19.0.1 rosbags==0.11.8 anthropic==1.15.0 numpy pillow scipy
# The agent stage also needs: $G/envs/sam3d/bin/pip install anthropic==1.15.0
set -euo pipefail
umask 077  # server.log and webapp.json hold the login token; job folders stay private
JOB=${1:?usage: start_ice.sh JOBID [PORT]}
PORT=${2:-0}
[[ $JOB =~ ^[0-9]+$ && $PORT =~ ^[0-9]+$ ]] || { echo "JOBID and PORT must be numbers" >&2; exit 2; }
G=$(readlink -f ~/scratch)/gator
PY=$G/envs/web/bin/python
INFO=~/.config/gator/webapp.json
LOG=$G/webapp_runs/server.log

say() { echo "$@" >&2; }
# "node port token" from webapp.json (the token stays off the command line), only if a server in job
# JOB on one of NODES wrote it
fields() {
  $PY -c 'import json, sys
d = json.load(open(sys.argv[1]))
if d.get("slurm_job") == sys.argv[2] and d.get("slurm_node") in sys.argv[3].split():
    print(d["node"], d["port"], d["token"])' "$INFO" "$JOB" "$NODES" 2>/dev/null
}
alive() {  # the server's token gate answers 401 to a request without the cookie
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$1:$2/api/jobs" || true)" = 401 ]
}

[ -x "$PY" ] || { say "no $PY; set up envs/web first (see the top of this script)"; exit 1; }
state=$(squeue -h -j "$JOB" -o '%T %N' 2>/dev/null || true)
[[ $state == RUNNING* ]] || { say "job $JOB is not running (squeue: ${state:-no such job})"; exit 1; }
# One name per line; squeue prints a list such as atl1-1-[02-03] for a job on more than one node.
NODES=$(scontrol show hostnames "${state#RUNNING }" 2>/dev/null || echo "${state#RUNNING }")

node= port= token=
if [ -s "$INFO" ] && read -r node port token < <(fields) && alive "$node" "$port"; then
  say "a server from an earlier start in job $JOB is still running on $node:$port; reusing it"
else
  node= port= token=
  mkdir -p "$G/webapp_runs"
  rm -f "$INFO"  # so that the file this start writes is the one read below
  # bash -l, so that module exists and purges what the login profile loaded; setsid -w keeps $!
  # alive as long as srun runs, also if setsid has to fork.
  nohup setsid -w srun --jobid="$JOB" --overlap --ntasks=1 \
    bash -lc "module purge 2>/dev/null; cd $G/gator_clone && exec $PY -m gator.webapp --port $PORT" \
    > "$LOG" 2>&1 < /dev/null &
  launcher=$!
  say "starting the server in job $JOB on ${state#RUNNING } (log: $LOG)"
  for _ in $(seq 90); do
    sleep 2
    [ -s "$INFO" ] && read -r node port token < <(fields) && break
    node=
    if ! kill -0 $launcher 2>/dev/null; then
      say "the server did not start:"; tail -20 "$LOG" >&2; exit 1
    fi
  done
  [ -n "$node" ] || { say "no $INFO of job $JOB after 180 s:"; tail -20 "$LOG" >&2; exit 1; }
  for _ in $(seq 30); do alive "$node" "$port" && break; sleep 1; done
  alive "$node" "$port" || say "warning: $node:$port does not answer yet; see $LOG"
  say "server running on $node:$port"
fi
say "forward: ssh -S ~/.ssh/cm-ice -O forward -L 8765:$node:$port ali497@login-ice.pace.gatech.edu"
say "then open: http://localhost:8765/?token=$token"
echo "NODE=$node"
echo "PORT=$port"
echo "TOKEN=$token"
