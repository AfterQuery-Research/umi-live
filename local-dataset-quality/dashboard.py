#!/usr/bin/env python3
"""Local copy of the collection dashboard with continuous-gate results.

Live view of the record -> upload -> process -> Hub
pipeline while collecting training data. Run from the repo root on the workstation:

    uv run --package umi-live-server python local-dataset-quality/dashboard.py
    # then open http://localhost:8900

Shows, refreshed every ~3 s: episodes on the Hub, episodes mid-pipeline on the server
(with their current stage), arms waiting for their pair mate, per-Pi upload backlog,
and a recent-events feed (paired / pushed / failures) so a fresh take can be watched
from stop-press to Hub. Read-only: touches nothing, safe to leave open while recording.
"""
import json
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _deployment_inventory() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / "provisioning" / "pair.env").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed = shlex.split(value, comments=True)
        if len(parsed) == 1:
            values[key.strip()] = parsed[0]
    return values


DEPLOYMENT = _deployment_inventory()
SERVER = DEPLOYMENT["SERVER"]
LEADER = DEPLOYMENT["LEADER"]
FOLLOWER = DEPLOYMENT["FOLLOWER"]
HOTSPOT_ENABLED = DEPLOYMENT.get("LEADER_HOTSPOT_ENABLED", "false").lower() == "true"
FOLLOWER_USER = FOLLOWER.split("@", 1)[0] if "@" in FOLLOWER else FOLLOWER
PIS = [
    ("umi1", LEADER, None),
    (
        "umi2",
        f"{FOLLOWER_USER}@10.42.0.2" if HOTSPOT_ENABLED else FOLLOWER,
        LEADER if HOTSPOT_ENABLED else None,
    ),
]
PORT = 8900

# Runs ON the box each poll: one JSON blob describing queue, stages, dataset, events.
BOX_PROBE = r'''cd ~/umi-live && python3 - <<'PY'
import json, subprocess, time
from pathlib import Path
env = dict(l.split("=", 1) for l in Path(".env").read_text().splitlines()
           if "=" in l and not l.startswith("#"))
data = Path(env.get("DATA_DIR", "data")).expanduser()
STAGES = ["umi1/decode", "umi2/decode", "umi1/glim", "umi2/glim", "gripper", "dataset"]
out = {"t": time.time(), "processing": [], "waiting_pair": [], "done": 0}
for ep in sorted(data.glob("raw_episodes/*")):
    if not ep.is_dir():
        continue
    if ep.name.startswith("paired_"):
        der = data / "derived" / ep.name
        stages_done = [s for s in STAGES if (der / s / "COMPLETE").is_file()]
        if len(stages_done) == len(STAGES):
            out["done"] += 1
        else:
            nxt = next(s for s in STAGES if s not in stages_done)
            out["processing"].append({"id": ep.name, "at": nxt, "done": len(stages_done), "of": len(STAGES)})
    else:
        out["waiting_pair"].append(ep.name)
info = data / "lerobot/local/umi-live/meta/info.json"
out["local_episodes"] = json.loads(info.read_text())["total_episodes"] if info.is_file() else 0
out["repo"] = env.get("HF_REPO_ID", "")
quality_path = Path.home() / "umi-live-local/dataset-quality/results/status.json"
try:
    out["quality"] = json.loads(quality_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    out["quality"] = {"watcher_error": f"status unavailable: {exc}"}
out["quality_service"] = subprocess.run(
    ["systemctl", "is-active", "umi-filter-quality"], capture_output=True, text=True
).stdout.strip()
log = subprocess.run(["journalctl", "-u", "umi-runner", "--since", "-30min", "--no-pager", "-o", "short-iso"],
                     capture_output=True, text=True).stdout
episode_ids = {
    path.name
    for folder in ("uploads", "raw_episodes", "derived")
    for path in (data / folder).glob("*")
}
events = []
for line in log.splitlines():
    if "libx264" in line:
        continue
    if any(ep_id in line for ep_id in episode_ids) and any(
        k in line for k in ("pair: merged", "pushed=", "FAILED", "skipped", "align()", "orphan")
    ):
        ts, _, msg = line.partition(" 192")
        events.append(ts[11:19] + " " + msg.split("]: ", 1)[-1][:110])
out["events"] = events[-8:]
print(json.dumps(out))
PY'''

PI_PROBE = ('echo "{\\"recorder\\": \\"$(systemctl is-active umi-recorder)\\", '
            '\\"pending\\": $(ls ~/umi-recordings/complete 2>/dev/null | wc -l), '
            '\\"sent\\": $(ls ~/umi-recordings/sent 2>/dev/null | wc -l)}"')

state = {"box": {}, "pis": {}, "hub": {}, "err": {}}
lock = threading.Lock()


def _ssh(host: str, cmd: str, timeout: int = 20, jump: str | None = None) -> str:
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
    if jump:
        args.extend(
            [
                "-J",
                jump,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
            ]
        )
    else:
        args.extend(["-o", "StrictHostKeyChecking=accept-new"])
    r = subprocess.run([*args, host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or f"ssh exited {r.returncode}").strip())
    return r.stdout.strip()


def poll_box():
    while True:
        try:
            blob = json.loads(_ssh(SERVER, BOX_PROBE))
            with lock:
                state["box"] = blob
                state["err"].pop("box", None)
        except Exception as e:
            with lock:
                state["err"]["box"] = str(e)[:80]
        time.sleep(3)


def poll_pis():
    while True:
        for name, host, jump in PIS:
            try:
                blob = json.loads(_ssh(host, PI_PROBE, timeout=12, jump=jump))
                with lock:
                    state["pis"][name] = blob
                    state["err"].pop(name, None)
            except Exception as e:
                with lock:
                    state["err"][name] = str(e)[:80]
        time.sleep(6)


def _hf_token() -> str | None:
    # The repo is created PRIVATE, so the poll needs auth: read the workstation
    # inventory (never committed, never sent anywhere).
    for inv in (Path.home() / "umi-live/provisioning/secrets",):
        try:
            for line in inv.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"') or None
        except OSError:
            pass
    return None


def _is_missing_repo(e: BaseException) -> bool:
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if "404" in str(e) or "Not Found" in str(e) or type(e).__name__ == "RepositoryNotFoundError":
            return True
        e = e.__cause__ or e.__context__
    return False


def poll_hub():
    from huggingface_hub import HfApi, hf_hub_download
    token = _hf_token()
    api = HfApi(token=token)
    while True:
        repo = state.get("box", {}).get("repo") or ""
        if repo:
            try:
                p = hf_hub_download(repo, "meta/info.json", repo_type="dataset",
                                    force_download=True, token=token)
                info = api.repo_info(repo, repo_type="dataset")
                meta = json.load(open(p))
                with lock:
                    state["hub"] = {
                        "episodes": meta["total_episodes"],
                        "minutes": round(meta["total_frames"] / max(meta.get("fps") or 30, 1) / 60, 1),
                        "last_push": info.last_modified.timestamp(),
                    }
                    state["err"].pop("hub", None)
            except Exception as e:
                with lock:
                    if _is_missing_repo(e):
                        state["hub"] = {"episodes": 0, "minutes": 0, "last_push": None}  # repo not created yet
                        state["err"].pop("hub", None)
                    else:
                        state["err"]["hub"] = str(e)[:80]
        time.sleep(20)


PAGE = """<html><head><title>UMI collection</title><meta charset="utf-8"><style>
body{background:#101014;color:#e8e8ee;font-family:ui-monospace,monospace;margin:2em}
.big{font-size:3.2em;font-weight:700}.lbl{color:#8a8a99;font-size:.85em}
.card{display:inline-block;background:#1a1a22;border-radius:10px;padding:1em 1.6em;margin:.4em;vertical-align:top;min-width:9em}
.ok{color:#6f6}.warn{color:#fc6}.bad{color:#f66}#events{background:#15151c;border-radius:10px;padding:1em;margin-top:1em;font-size:.85em;line-height:1.7}
.quality{background:#15151c;border-radius:10px;padding:1.2em;margin-top:1em}
.quality-head{display:flex;align-items:end;justify-content:space-between;gap:1em;flex-wrap:wrap}
.quality-count{font-size:2.4em;font-weight:700;color:#6f6}.quality-sub{color:#8a8a99;font-size:.85em}
.episode-grid{display:grid;grid-template-rows:repeat(10,13px);grid-auto-flow:column;grid-auto-columns:13px;gap:4px;margin:1em 0;overflow-x:auto;padding:2px 0 7px}
.episode-cell{width:13px;height:13px;border-radius:3px;border:1px solid #3b3b45;background:#30303a}
.episode-cell.pass{background:#39d353;border-color:#39d353}.episode-cell.pass.optimized{background:#58a6ff;border-color:#58a6ff}.episode-cell.fail{background:#fff;border-color:#d9d9df}
.episode-cell.processing{background:#d29922;border-color:#d29922;animation:pulse 1.3s infinite alternate}
.episode-cell.error{background:#f66;border-color:#f66}@keyframes pulse{to{opacity:.45}}
.legend{display:flex;gap:1em;flex-wrap:wrap;color:#8a8a99;font-size:.78em}.key{display:inline-flex;align-items:center;gap:.35em}.swatch{width:10px;height:10px;border-radius:2px;border:1px solid #555}
.failed-dropdown{margin-top:.8em}.failed-dropdown summary{cursor:pointer;color:#ddd}.failed-list{display:flex;flex-wrap:wrap;gap:.4em;margin-top:.7em}.failed-chip{background:#fff;color:#17171d;border-radius:5px;padding:.25em .5em;font-size:.82em}
</style></head><body>
<h2 id="repo">…</h2><div id="cards">loading…</div>
<section class="quality">
  <div class="quality-head"><div><div class="quality-count" id="quality-count">—</div><div>episodes pass the continuous optimization gate</div></div><div class="quality-sub" id="quality-meta">waiting for Lambda…</div></div>
  <div class="episode-grid" id="episode-grid" aria-label="Continuous optimization results"></div>
  <div class="legend"><span class="key"><i class="swatch" style="background:#39d353;border-color:#39d353"></i>standard pass</span><span class="key"><i class="swatch" style="background:#58a6ff;border-color:#58a6ff"></i>pass with optimization</span><span class="key"><i class="swatch" style="background:#fff"></i>fail</span><span class="key"><i class="swatch" style="background:#d29922;border-color:#d29922"></i>processing</span><span class="key"><i class="swatch" style="background:#30303a"></i>pending</span></div>
  <details class="failed-dropdown"><summary id="failed-summary">Failed episodes</summary><div class="failed-list" id="failed-list"></div></details>
</section>
<div id="events"></div>
<script>
async function tick(){
  const s = await (await fetch('/status.json')).json();
  const b = s.box||{}, hub = s.hub||{};
  document.getElementById('repo').textContent = b.repo || 'connecting to server…';
  const age = hub.last_push ? Math.round(Date.now()/1000 - hub.last_push) : null;
  const ageTxt = age===null ? '—' : age<90 ? age+'s ago' : Math.round(age/60)+'m ago';
  const justSent = age!==null && age<90;
  let cards = '';
  cards += `<div class=card><div class=big>${hub.episodes ?? '—'}</div><div class=lbl>episodes on the Hub</div><div class="lbl ${justSent?'ok':''}">last push ${ageTxt}${justSent?' ✓':''}</div></div>`;
  cards += `<div class=card><div class=big>${hub.minutes ?? '—'}</div><div class=lbl>minutes of data on the Hub</div></div>`;
  cards += `<div class=card><div class=big class=warn>${(b.processing||[]).length}</div><div class=lbl>processing now</div><div class=lbl>${(b.processing||[]).map(p=>p.id.slice(-6)+' @ '+p.at+' ('+p.done+'/'+p.of+')').join('<br>')||'idle'}</div></div>`;
  cards += `<div class=card><div class=big>${(b.waiting_pair||[]).length}</div><div class=lbl>arms waiting for pair mate</div></div>`;
  for (const [name,pi] of Object.entries(s.pis||{})) {
    const okc = pi.recorder==='active' ? 'ok' : 'bad';
    cards += `<div class=card><div class="big ${pi.pending>0?'warn':''}">${pi.pending}</div><div class=lbl>${name} uploads queued</div><div class="lbl ${okc}">recorder ${pi.recorder} · ${pi.sent} sent</div></div>`;
  }
  const errs = Object.entries(s.err||{}).map(([k,v])=>`<span class=bad>${k}: ${v}</span>`).join(' · ');
  document.getElementById('cards').innerHTML = cards + (errs?`<div class=lbl>${errs}</div>`:'');
  const q = b.quality || {};
  const optimizedPasses = (q.episodes || []).filter(ep => ep.state === 'pass' && ep.selected_profile && ep.selected_profile !== 'baseline_v3').length;
  document.getElementById('quality-count').textContent = q.passed ?? '—';
  const service = b.quality_service || 'unknown';
  document.getElementById('quality-meta').textContent = q.watcher_error
    ? `${service} · ${q.watcher_error}`
    : q.waiting_for_first_episode
    ? `Waiting for the first processed episode · ${q.workers ?? '—'} Lambda CPU workers ready`
    : `${q.evaluated ?? 0} of ${q.dataset_total ?? 0} evaluated · ${optimizedPasses} optimized passes · ${q.processing ?? 0} processing · ${q.workers ?? '—'} Lambda CPU workers`;
  const grid = document.getElementById('episode-grid');
  grid.replaceChildren(...(q.episodes || []).map(ep => {
    const cell = document.createElement('span');
    const optimized = ep.state === 'pass' && ep.selected_profile && ep.selected_profile !== 'baseline_v3';
    cell.className = `episode-cell ${ep.state}${optimized ? ' optimized' : ''}`;
    const reasons = (ep.failure_reasons || []).join(', ');
    const passLabel = optimized ? `pass with ${ep.selected_profile}` : ep.state;
    cell.title = `episode ${ep.episode_index}: ${passLabel}${reasons ? ' · '+reasons : ''}${ep.error ? ' · '+ep.error : ''}`;
    cell.setAttribute('aria-label', cell.title);
    return cell;
  }));
  const failed = q.failed_episode_indices || [];
  document.getElementById('failed-summary').textContent = `Failed episodes (${failed.length})`;
  const failedList = document.getElementById('failed-list');
  failedList.replaceChildren(...failed.map(index => {
    const chip = document.createElement('span');
    chip.className = 'failed-chip';
    chip.textContent = index;
    return chip;
  }));
  if (!failed.length) failedList.textContent = q.evaluated ? 'No failed episodes.' : 'No evaluated episodes yet.';
  document.getElementById('events').innerHTML = '<b>recent pipeline events</b><br>' + (b.events||[]).slice().reverse().join('<br>');
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/status.json":
            with lock:
                body = json.dumps(state).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


def main():
    for fn in (poll_box, poll_pis, poll_hub):
        threading.Thread(target=fn, daemon=True).start()
    print(f"serving http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
