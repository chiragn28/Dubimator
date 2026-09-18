# Hosting Dubimator on a server

The landing page and the architecture page are static and live on Vercel. The rest of the
platform — the API, the Streamlit demo, Postgres with the 1,047,965 transactions, and the MLflow
model store — needs a machine that stays on. This directory holds everything needed to move that
stack off a laptop and onto one small Linux server.

Nothing here needs a GPU. Training used CUDA; serving is CPU-only.

## What you need

| | |
|---|---|
| Server | 4 vCPU, **8 GB RAM**, 80 GB disk, Ubuntu 24.04. Hetzner CX32 (~€8/month) or equivalent. 4 GB is not enough: the API holds the price model, the search ranker with CPU torch, and the sentence-transformer embedder. |
| Domain | Two A records at the server's IP, e.g. `api.example.com` and `demo.example.com` (~$10/year). |
| On your PC | Docker Desktop running, this repository, and a working `.env`. |

## Steps

**1. Create the server** with your SSH key, and point both DNS records at its IP. Wait for
`ping api.example.com` to answer from the right address before step 4 — Caddy needs DNS to
resolve to issue certificates.

**2. Clone the repository on the server**

```bash
ssh root@SERVER_IP
git clone https://github.com/chiragn28/Dubimator.git /opt/dubimator
```

**3. Ship the data from your PC** (PowerShell, from the repository root). This stops the local
stack, exports the two volumes that hold real state, and copies them plus `.env` over:

```powershell
.\scripts\export_volumes.ps1 -Server root@SERVER_IP
```

About 1.2 GB. The `hf_cache` volume is not copied — the API re-downloads the embedding model on
first start. Start your local stack again with `docker compose up -d` when it finishes.

**4. Run the setup script on the server**

```bash
cd /opt/dubimator && bash deploy/server_setup.sh api.example.com demo.example.com
```

It installs Docker and Caddy, restores the volumes, points Caddy at the two domains, enables the
firewall (22, 80, 443 only), builds the images and starts everything. The first build takes
10–15 minutes because the API image installs CPU torch.

**5. Rotate the secrets.** The `.env` you copied has your laptop's passwords in it. On the
server, edit `/opt/dubimator/.env`:

```bash
openssl rand -hex 24   # once per key below
```

- `POSTGRES_PASSWORD` — then apply it to the running database:
  `docker exec -it dubimator-postgres psql -U dubimator -c "ALTER USER dubimator PASSWORD 'new';"`
- `API_KEYS` — a comma-separated list; `DEMO_API_KEY`, `PROMETHEUS_API_KEY` and every entry of
  `API_ADMIN_KEYS` must each also appear in `API_KEYS`.
- `GRAFANA_ADMIN_PASSWORD`.
- `API_CORS_ORIGINS=https://dubimator.vercel.app,https://demo.example.com`.

Then `docker compose up -d --force-recreate api demo` and re-check `/v1/ready`.

**6. Point the landing page at the live demo.** On your PC, in `landing/`:

```powershell
$env:VITE_DEMO_URL = "https://demo.example.com"; npm run build
vercel deploy dist --prod --yes --name dubimator
```

## Verify

```bash
curl -s https://api.example.com/v1/ready     # every component "up"
curl -s https://api.example.com/v1/areas -H "X-API-Key: <one of API_KEYS>" | head -c 200
```

Open `https://demo.example.com` and run one price estimate. Then reboot the server once
(`reboot`): every service has `restart: unless-stopped`, so the whole stack should come back on
its own.

## What is deliberately not public

Only the API and the demo are reachable from the internet. Airflow, MLflow, Prometheus and
Grafana stay bound to `127.0.0.1` — reach them over an SSH tunnel:

```bash
ssh -L 3000:127.0.0.1:3000 -L 5000:127.0.0.1:5000 root@SERVER_IP
# then http://127.0.0.1:3000 (Grafana) and http://127.0.0.1:5000 (MLflow)
```

## Running costs

About €8–9/month for the server plus the domain. Vercel's free tier covers the two static pages.

## Retraining

`python -m pipelines retrain` is CPU-only but slow on 4 vCPU, so leave the Airflow schedule off
on the server. Retrain locally on the GPU instead, then re-sync the model store:

```powershell
docker run --rm -v dubimator_mlflow_data:/v:ro -v ${PWD}:/b alpine tar czf /b/mlflow_data.tgz -C /v .
scp mlflow_data.tgz root@SERVER_IP:/opt/dubimator/
```

On the server, stop the API, restore the volume as in `server_setup.sh`, and start it again.

## Cheaper and more expensive alternatives

- **Cloudflare Tunnel from your PC** (free): the `tunnel` and `tunnel-named` compose profiles are
  already wired up, and `scripts/tunnel_url.py` prints the quick-tunnel URL. Only up while the PC
  is on, which is why it is not the recommendation here.
- **Managed pieces:** Streamlit Community Cloud for the demo, Neon or Supabase for Postgres, and
  the API on Fly.io or Railway. More dashboards to manage, and the 1.2 GB database is over most
  free Postgres tiers.
