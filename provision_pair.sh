#!/usr/bin/env bash
# provision_pair.sh <leader user@host> <follower user@host>
#
# Provision a coordinated two-Pi BIMANUAL pair by running provision_pi.sh for each Pi.
# Each Pi needs its own provisioning/rigs/<host>.env (see provisioning/rigs/example.env):
#   - the FOLLOWER's env: ROLE="follower", a umi2-native RIG_CONFIG_FILE
#     (config/umi.pair-follower.example.yaml), TIME_REF="<shared-ntp-ip>"
#   - the LEADER's env:   ROLE="leader",  a umi1 RIG_CONFIG_FILE
#     (config/umi.pair-leader.example.yaml), PEER_HOST="<follower-ip>", TIME_REF="<same-ntp-ip>"
#
# The follower is provisioned first (so it's listening before the leader comes up), though
# the leader retries the control channel at runtime, so order isn't load-bearing. On the
# server, run the runner with --pair so the two single-arm episodes merge into one 12-D
# bimanual dataset (server pair_stage).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LEADER="${1:?usage: provision_pair.sh <leader user@host> <follower user@host>}"
FOLLOWER="${2:?usage: provision_pair.sh <leader user@host> <follower user@host>}"

echo "=== provisioning FOLLOWER  $FOLLOWER  (headless, records the umi2 arm) ==="
"$HERE/provision_pi.sh" "$FOLLOWER"

echo
echo "=== provisioning LEADER  $LEADER  (buttons + LCD, records the umi1 arm) ==="
"$HERE/provision_pi.sh" "$LEADER"

echo
echo "pair provisioned. Press record on the LEADER — both Pis record the same take;"
echo "each uploads its single-arm episode; the server (runner --pair) merges them into"
echo "one 12-D bimanual dataset. Verify cross-Pi sync with 'chronyc tracking' on both."
