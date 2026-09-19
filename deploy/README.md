# Hosting Dubimator on the web

The landing page and the architecture page are static and already live on Vercel. The rest of the
platform — the API, the Streamlit demo, Postgres with the 1,047,965 transactions, and the MLflow
model store — needs a machine that stays on. This is the walkthrough for putting that on one
small Linux server, from nothing to a live HTTPS URL.

**Time:** about 45 minutes, most of it waiting for a Docker build.
**Cost:** about €9/month for the server, about $10/year for a domain.
**You need:** a card for the server, and Docker Desktop running on your PC for step 4.

Nothing here needs a GPU. Training used CUDA; serving is CPU-only.

---

## Step 0 — Check your SSH key

On your PC, in PowerShell:

```powershell
cat ~/.ssh/id_ed25519.pub
```

If that prints a line starting `ssh-ed25519`, you have a key — copy the whole line, you'll paste
it in step 1. If it errors, create one first (press Enter at every prompt):

```powershell
ssh-keygen -t ed25519
```

---

## Step 1 — Create the server

Any provider works; these are the clicks for [Hetzner Cloud](https://console.hetzner.cloud),
which is the cheapest of the big ones.

1. Sign up, then **New project** → name it `dubimator` → open it.
2. **Add Server**.
3. **Location:** Nuremberg or Falkenstein (Germany), or Ashburn if you're in the US.
4. **Image:** Ubuntu 24.04.
5. **Type:** Shared vCPU → **CX32** (4 vCPU, 8 GB RAM, 80 GB disk, ~€8.50/month).
   **Do not pick a 4 GB plan.** The API holds the price model, the search ranker with CPU torch
   and the sentence-transformer embedder; 4 GB runs out and the container is killed.
6. **SSH keys:** **Add SSH key** → paste the line from step 0 → give it any name.
7. **Name:** `dubimator` → **Create & Buy now**.

After about 30 seconds the server has an IPv4 address. Copy it.

**Checkpoint.** From your PC:

```powershell
ssh root@YOUR_SERVER_IP
```

You should get a root shell without being asked for a password. Type `exit` to come back.

---

## Step 2 — Point a domain at it

Buy a domain anywhere (Namecheap, Porkbun, Cloudflare — about $10/year). In its DNS settings add
**two A records**, both pointing at your server's IP:

| Type | Name | Value |
|---|---|---|
| A | `api` | YOUR_SERVER_IP |
| A | `demo` | YOUR_SERVER_IP |

That gives you `api.yourdomain.com` and `demo.yourdomain.com`.

**Checkpoint.** Wait a few minutes, then:

```powershell
nslookup api.yourdomain.com
```

It must answer with your server's IP before you run step 5 — Caddy can't get an HTTPS certificate
until DNS resolves. If it still shows the old or no address, wait and try again.

> **No domain?** You can try `api.YOUR-IP-WITH-DASHES.nip.io` (e.g. `api.203-0-113-10.nip.io`),
> which resolves to your IP with nothing to buy. It usually works, but shared free domains
> sometimes hit certificate rate limits, so a real domain is the reliable path.

---

## Step 3 — Put the code on the server

```powershell
ssh root@YOUR_SERVER_IP
```

Then, on the server:

```bash
apt update && apt install -y git
git clone https://github.com/chiragn28/Dubimator.git /opt/dubimator
exit
```

---

## Step 4 — Send the database and models from your PC

The code comes from GitHub, but the 1M transactions and the trained models only exist in Docker
volumes on your PC. One script moves them. **Start Docker Desktop first**, then in PowerShell
from the repository folder:

```powershell
.\scripts\export_volumes.ps1 -Server root@YOUR_SERVER_IP
```

It stops your local stack, packs the two volumes (about 1.2 GB), and copies them plus your `.env`
to the server. Expect a few minutes on a slow upload. When it finishes, bring your local stack
back up if you want it:

```powershell
docker compose up -d
```

**Checkpoint.** The script prints the two file sizes before copying. `postgres_data.tgz` should be
a few hundred MB, not a few KB.

---

## Step 5 — Run the setup script

Back on the server, with **your** two domains:

```bash
ssh root@YOUR_SERVER_IP
cd /opt/dubimator
bash deploy/server_setup.sh api.yourdomain.com demo.yourdomain.com
```

It installs Docker and Caddy, restores the two volumes, writes the HTTPS config for your domains,
opens only ports 22/80/443, builds the images and starts everything. **The build takes 10–15
minutes** — the API image installs CPU torch, which is large. Leave it running.

**Checkpoint.** It ends by printing `docker compose ps`; every service should say `running` or
`healthy`. Then, from anywhere:

```bash
curl -s https://api.yourdomain.com/health
```

That is the one public endpoint and should return `{"status":"ok"}` over HTTPS. The detailed
readiness check needs a key, so it comes in step 6.

---

## Step 6 — Change the passwords

The `.env` you copied in step 4 holds your laptop's passwords. Replace them now. On the server:

```bash
nano /opt/dubimator/.env
```

Generate each value with `openssl rand -hex 24` (run it once per key) and set:

- `POSTGRES_PASSWORD` — then apply it to the running database:
  ```bash
  docker exec -it dubimator-postgres psql -U dubimator -c "ALTER USER dubimator PASSWORD 'the-new-one';"
  ```
- `API_KEYS` — a comma-separated list of new keys. Important: `DEMO_API_KEY`,
  `PROMETHEUS_API_KEY` and every entry of `API_ADMIN_KEYS` must **also** appear in `API_KEYS`, or
  the API rejects them.
- `GRAFANA_ADMIN_PASSWORD` — any new value.
- `API_CORS_ORIGINS=https://dubimator.vercel.app,https://demo.yourdomain.com`

Then restart and re-check:

```bash
cd /opt/dubimator && docker compose up -d --force-recreate api demo
curl -s https://api.yourdomain.com/v1/ready -H "X-API-Key: ONE_OF_YOUR_NEW_KEYS"
```

Every component should be `"up"`. `/v1/ready` and every `/v1/...` route needs the key header;
only `/health` is open.

---

## Step 7 — Point the landing page at the live demo

On your PC, in the `landing` folder:

```powershell
cd landing
$env:VITE_DEMO_URL = "https://demo.yourdomain.com"
npm run build
vercel deploy dist --prod --yes --name dubimator
```

Now "Try the Demo" and the four nav links on https://dubimator.vercel.app go to your live demo
instead of `localhost`.

To make that permanent for the automatic git deploys, set `VITE_DEMO_URL` in the Vercel
dashboard: **Project `dubimator` → Settings → Environment Variables → Add**, name
`VITE_DEMO_URL`, value `https://demo.yourdomain.com`, all environments.

---

## Step 8 — Final check

```bash
curl -s https://api.yourdomain.com/health                                     # {"status":"ok"}
curl -s https://api.yourdomain.com/v1/ready -H "X-API-Key: YOUR_KEY"          # all components up
curl -s "https://api.yourdomain.com/v1/areas" -H "X-API-Key: YOUR_KEY" | head -c 200
```

Open `https://demo.yourdomain.com` and run one price estimate end to end. Then reboot the server
once to prove it comes back by itself:

```bash
reboot
```

Wait a minute, then run the two checks again. Every service has `restart: unless-stopped`, so the
whole stack should return without you touching it.

---

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| Browser says the certificate is invalid, or Caddy logs `no such host` | DNS isn't pointing at the server yet. Check `nslookup api.yourdomain.com`, wait, then `systemctl reload caddy`. |
| `/v1/ready` returns 401 | You didn't send the key, or it isn't in `API_KEYS`. Add `-H "X-API-Key: ..."`. |
| `/v1/ready` shows a component `down` | `docker compose logs api --tail 50`. Usually a missing key in `.env` or the model store didn't restore — check `docker run --rm -v dubimator_mlflow_data:/v alpine ls /v` is not empty. |
| The API container keeps restarting | Out of memory — confirm you're on an 8 GB plan with `free -h`. |
| `server_setup.sh` says `postgres_data.tgz missing` | Step 4 didn't complete. Re-run `export_volumes.ps1` with Docker Desktop running. |
| Build fails downloading packages | Transient; just run `bash deploy/server_setup.sh api.yourdomain.com demo.yourdomain.com` again. It skips what already succeeded. |
| `401` from the API | The key you're sending isn't in `API_KEYS`. Send it as the `X-API-Key` header. |

---

## What is deliberately not public

Only the API and the demo are reachable from the internet. Airflow, MLflow, Prometheus and
Grafana stay bound to `127.0.0.1`. Reach them through an SSH tunnel from your PC:

```powershell
ssh -L 3000:127.0.0.1:3000 -L 5000:127.0.0.1:5000 -L 8080:127.0.0.1:8080 root@YOUR_SERVER_IP
```

Then open `http://127.0.0.1:3000` (Grafana), `http://127.0.0.1:5000` (MLflow) and
`http://127.0.0.1:8080` (Airflow) in your browser while that session is open.

---

## Automatic deploys

Every push to `master` redeploys the server once CI passes. The workflow is
`.github/workflows/deploy.yml`; it logs in over SSH and runs `deploy/redeploy.sh`, which pulls
the new code, rebuilds only what changed, and fails (turning the Action red) if the API does not
come back with every component up. The landing page is separate: Vercel deploys it from the repo.

The SSH key it uses is not your personal one. Its public half sits in the server's
`/root/.ssh/authorized_keys` with `restrict,command="bash /opt/dubimator/deploy/redeploy.sh"` in
front, so that key can run the redeploy script and nothing else. Its private half is the
repository secret `DEPLOY_SSH_KEY` (GitHub, Settings, Secrets and variables, Actions).

To rotate it: `ssh-keygen -t ed25519 -N "" -f k`, replace the `github-actions-deploy` line in
`authorized_keys` with the new public key (keep the `restrict,command=...` prefix), and update the
secret. To deploy by hand, run `ssh root@SERVER 'bash /opt/dubimator/deploy/redeploy.sh'`, or use
"Run workflow" on the Deploy action.

---

## Keeping it running

**Updating the code.** Push to `master`, then on the server:

```bash
cd /opt/dubimator && git pull && docker compose up -d --build api demo
```

**Retraining.** `python -m pipelines retrain` is CPU-only but slow on 4 vCPU, so leave Airflow's
schedule off on the server. Retrain locally on your GPU, then re-send just the model store:

```powershell
docker run --rm -v dubimator_mlflow_data:/v:ro -v ${PWD}:/b alpine tar czf /b/mlflow_data.tgz -C /v .
scp mlflow_data.tgz root@YOUR_SERVER_IP:/opt/dubimator/
```

On the server, stop the API, delete and restore the volume the way `server_setup.sh` does, and
start it again.

**Backups.** Hetzner's automatic backups are about €1.70/month on this plan — worth it, and the
simplest way to protect the database.

---

## Cheaper and more expensive alternatives

- **Cloudflare Tunnel from your PC** (free, no server, no domain): the `tunnel` and
  `tunnel-named` compose profiles are already wired up, and `scripts/tunnel_url.py` prints the
  quick-tunnel URL. The link only works while your PC is on, which is why it isn't the
  recommendation here.
- **Managed pieces:** Streamlit Community Cloud for the demo, Neon or Supabase for Postgres, the
  API on Fly.io or Railway. More dashboards to manage, and the 1.2 GB database is over most free
  Postgres tiers.
