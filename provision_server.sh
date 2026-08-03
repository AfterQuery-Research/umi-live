#!/usr/bin/env bash
# provision_server.sh — stand up the PROCESSING HOST from your workstation, the same
# way provision_pi.sh stands up a Pi: one local inventory, rendered and pushed. The
# upload token is written ONCE (provisioning/secrets) and fanned out to the box here
# and to each Pi by provision_pi.sh — the two can never mismatch.
#
#   cp provisioning/secrets.example    provisioning/secrets      # 3 secrets
#   cp provisioning/server.env.example provisioning/server.env   # box settings
#   ./provision_server.sh ubuntu@<box-ip>     # remote GPU box: render .env, rsync repo,
#                                             # run install_server.sh, verify
#   ./provision_server.sh --local             # Mac-as-host: just render ./.env here,
#                                             # then run: PAIR=1 mac/run_processing.sh
#
# Idempotent; re-run after editing the inventory (edit the inventory, not the box).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "provision_server: $*" >&2; exit 1; }
log() { echo; echo "=== $* ==="; }

BOX=""; LOCAL=0
case "${1:-}" in
  --local) LOCAL=1;;
  -h|--help|"") sed -n '2,15p' "$0"; exit 0;;
  *@*) BOX="$1";;
  *) die "usage: ./provision_server.sh <user@box> | --local";;
esac

SECRETS="$HERE/provisioning/secrets"
SRVENV="$HERE/provisioning/server.env"
[ -f "$SECRETS" ] || die "missing $SECRETS  (cp provisioning/secrets.example provisioning/secrets)"
[ -f "$SRVENV" ] || die "missing $SRVENV  (cp provisioning/server.env.example provisioning/server.env)"
# shellcheck disable=SC1090
. "$SECRETS"
# shellcheck disable=SC1090
. "$SRVENV"
[ -n "${UPLOAD_TOKEN:-}" ] || die "UPLOAD_TOKEN is blank in $SECRETS  (openssl rand -hex 32)"
[ -n "${REPO_ID:-}" ] || die "REPO_ID is blank in $SRVENV  (e.g. yourname/umi-dataset)"
[ -n "${HF_TOKEN:-}" ] || echo "provision_server: WARN HF_TOKEN blank — datasets stay local-only (no Hub push)"

render_env() {  # $1 = DATA_DIR value to embed
  cat <<ENV
# Rendered by provision_server.sh from provisioning/secrets + provisioning/server.env.
# Edit the inventory and re-run ./provision_server.sh — not this file.
ROBOT_UMI_UPLOAD_TOKEN=$UPLOAD_TOKEN
HF_TOKEN=${HF_TOKEN:-}
ROBOT_UMI_REPO_ID=$REPO_ID
ROBOT_UMI_PUSH_TO_HUB=${PUSH_TO_HUB:-1}
ROBOT_UMI_PUBLIC=${PUBLIC:-0}
ROBOT_UMI_DATASET_ROOT=data/lerobot
ROBOT_UMI_ENGINE=glim
DATA_DIR=$1
ROBOT_UMI_RUNNER_ARGS=${RUNNER_ARGS:-}
ENV
}

if [ "$LOCAL" = 1 ]; then
  # Mac-as-host: the processing host IS this machine; render the repo-local .env that
  # mac/run_processing.sh sources. (HOST_UID/GID are a Linux-compose concern only.)
  render_env "./data" > "$HERE/.env"
  echo "provision_server: wrote $HERE/.env (token/HF/repo from the inventory)."
  echo "Start processing with:  PAIR=1 INGEST_HOST=0.0.0.0 mac/run_processing.sh"
  exit 0
fi

BOX_USER="${BOX%@*}"
SSHOPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes)
bssh() { ssh "${SSHOPTS[@]}" "$BOX" "$@"; }

log "1/4 preflight"
bssh true || die "cannot SSH to $BOX — set up key access first"
bssh 'sudo -n true' 2>/dev/null \
  || die "$BOX needs passwordless sudo for install_server.sh (standard on cloud images)"
REMOTE_REPO="/home/$BOX_USER/$(basename "$HERE")"

log "2/4 rsync repo -> $BOX:$REMOTE_REPO"
# The inventory (secrets, rig files) and local data/venv NEVER leave this machine;
# the box gets code + a rendered .env only.
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'data' --exclude '__pycache__' \
  --exclude 'provisioning' --exclude '.env' --exclude 'config/umi.yaml' \
  --exclude 'device/v4l2_recorder/target' --exclude '*.rrd' \
  --exclude 'docker-compose.override.yml' \
  -e "ssh ${SSHOPTS[*]}" "$HERE/" "$BOX:$REMOTE_REPO/"

log "3/4 render + push .env"
# Preserve installer-managed HOST_UID/GID lines across re-provisions if present.
KEEP="$(bssh "grep -E '^HOST_[UG]ID=' $REMOTE_REPO/.env 2>/dev/null" || true)"
{ render_env "$REMOTE_REPO/data"; [ -n "$KEEP" ] && printf '%s\n' "$KEEP"; } \
  | bssh "cat > $REMOTE_REPO/.env"
echo "server .env rendered (token fanned out from provisioning/secrets)"

log "4/4 install_server.sh on $BOX (idempotent; first run ~15 min)"
bssh "cd $REMOTE_REPO && bash server/scripts/install_server.sh" \
  || die "install_server.sh failed on $BOX (see output above)"

echo
echo "provision_server: done. Verify:"
echo "  ssh $BOX 'curl -s -o /dev/null -w \"%{http_code}\\n\" -X POST http://localhost:8080/upload'   # 401 = up"
echo "Now provision the Pis against it:  ./provision_pi.sh --target gpu <user@pi>"
echo "(rig env: SERVER_SSH=\"$BOX\" for an SSH-only cloud box, or SERVER_URL for a LAN box)"
