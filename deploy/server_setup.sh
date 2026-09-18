#!/usr/bin/env bash
# One-shot server setup for Dubimator on a fresh Ubuntu 24.04 box.
#
#   bash deploy/server_setup.sh api.example.com demo.example.com
#
# Run it from the clone (/opt/dubimator) as root, after scripts/export_volumes.ps1 has copied
# .env, postgres_data.tgz and mlflow_data.tgz here. Safe to re-run: it skips what already exists.
set -euo pipefail

API_DOMAIN=${1:-}
DEMO_DOMAIN=${2:-}
if [[ -z $API_DOMAIN || -z $DEMO_DOMAIN ]]; then
  echo "usage: bash deploy/server_setup.sh <api domain> <demo domain>" >&2
  exit 64
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
[[ -f .env ]] || { echo "No .env here. Run scripts/export_volumes.ps1 from your PC first." >&2; exit 65; }

echo "==> Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg git ufw debian-keyring debian-archive-keyring apt-transport-https

if ! command -v docker >/dev/null; then
  echo "==> Docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! command -v caddy >/dev/null; then
  echo "==> Caddy"
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
fi

echo "==> Restoring volumes"
for v in postgres_data mlflow_data; do
  if docker volume inspect "dubimator_$v" >/dev/null 2>&1 && [[ -n $(docker run --rm -v "dubimator_$v:/v" alpine sh -c 'ls -A /v') ]]; then
    echo "    dubimator_$v already has data, leaving it alone"
    continue
  fi
  [[ -f $v.tgz ]] || { echo "    $v.tgz missing - run scripts/export_volumes.ps1 from your PC" >&2; exit 66; }
  docker volume create "dubimator_$v" >/dev/null
  docker run --rm -v "dubimator_$v:/v" -v "$REPO:/b:ro" alpine tar xzf "/b/$v.tgz" -C /v
  echo "    restored dubimator_$v"
done

echo "==> Compose overrides"
# So plain `docker compose ...` picks up the production overrides on this host.
grep -q '^COMPOSE_FILE=' .env || echo 'COMPOSE_FILE=docker-compose.yml:deploy/docker-compose.prod.yml' >> .env
grep -q '^COMPOSE_PROFILES=' .env || echo 'COMPOSE_PROFILES=monitoring' >> .env

echo "==> Caddy config for $API_DOMAIN and $DEMO_DOMAIN"
sed -e "s/API_DOMAIN/$API_DOMAIN/" -e "s/DEMO_DOMAIN/$DEMO_DOMAIN/" deploy/Caddyfile > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

echo "==> Firewall"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null

echo "==> Building images (10-15 minutes the first time)"
docker compose build api demo mlflow

echo "==> Starting"
docker compose up -d
docker compose ps

echo
echo "Now check:"
echo "  curl -s https://$API_DOMAIN/v1/ready"
echo "  open  https://$DEMO_DOMAIN"
echo
echo "If .env still has the passwords from your PC, rotate them now - see deploy/README.md."
