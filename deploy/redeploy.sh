#!/usr/bin/env bash
# Pull the latest master and restart whatever changed. Runs on the server.
#
# Two callers:
#   - GitHub Actions (.github/workflows/deploy.yml), over an SSH key that is locked to this one
#     script in /root/.ssh/authorized_keys, so the key can do nothing else on the server.
#   - You, by hand:  ssh root@SERVER 'bash /opt/dubimator/deploy/redeploy.sh'
#
# It fails loudly (non-zero exit) if the new version does not come up healthy, which turns the
# GitHub Action red instead of leaving a half-working site unnoticed.
set -euo pipefail

cd /opt/dubimator

# Two pushes in quick succession must not build on top of each other.
exec 9>/var/lock/dubimator-deploy.lock
flock -w 1200 9 || { echo "another deploy is still running" >&2; exit 75; }

echo "==> Pull"
git fetch --quiet origin master
# --ff-only: if someone edited tracked files on the server, stop rather than overwrite them.
git merge --ff-only origin/master
echo "    now at $(git log --oneline -1)"

echo "==> Build and restart (only what changed is rebuilt)"
docker compose up -d --build postgres mlflow api demo

echo "==> Wait for the API to load its models"
key=$(grep '^DEMO_API_KEY=' .env | cut -d= -f2-)
healthy=""
for _ in $(seq 1 90); do
  if ready=$(curl -fsS -m 5 http://127.0.0.1:8000/v1/ready -H "X-API-Key: $key" 2>/dev/null) \
     && python3 -c 'import json,sys; c=json.load(sys.stdin)["components"]; sys.exit(0 if c and all(v.get("up") for v in c.values()) else 1)' <<<"$ready" \
     && curl -fsS -m 5 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 4
done

if [[ -z $healthy ]]; then
  echo "!! The new version did not become healthy within 6 minutes" >&2
  docker compose ps >&2
  docker compose logs --tail 40 api demo >&2
  exit 1
fi

docker image prune -f >/dev/null
echo "==> Deployed $(git rev-parse --short HEAD): every API component is up"
