#!/usr/bin/env python3
"""Episode-collection dashboard: live view of the record -> upload -> process -> Hub
pipeline while collecting training data. Run from the repo root on the workstation:

    uv run --package umi-flow-server python tools/collect_dashboard.py
    # then open http://localhost:8900

Shows, refreshed every ~3 s: episodes on the Hub, episodes mid-pipeline on the server
(with their current stage), arms waiting for their pair mate, per-Pi upload backlog,
and a recent-events feed (paired / pushed / failures) so a fresh take can be watched
from stop-press to Hub. Read-only: touches nothing, safe to leave open while recording.
"""
import json
import subprocess
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER = "ubuntu@192.222.52.98"
PIS = [("umi1", "umi1@umi1.local"), ("umi2", "umi2@umi2.local")]
PORT = 8900

# Runs ON the box each poll: one JSON blob describing queue, stages, dataset, events.
BOX_PROBE = r'''cd ~/umi-flow && python3 - <<'PY'
import json, subprocess, time
from pathlib import Path
data = Path("data")
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
info = data / "lerobot/local/umi-flow/meta/info.json"
out["local_episodes"] = json.loads(info.read_text())["total_episodes"] if info.is_file() else 0
env = dict(l.split("=", 1) for l in Path(".env").read_text().splitlines() if "=" in l and not l.startswith("#"))
out["repo"] = env.get("HF_REPO_ID", "")
log = subprocess.run(["journalctl", "-u", "umi-runner", "--since", "-30min", "--no-pager", "-o", "short-iso"],
                     capture_output=True, text=True).stdout
events = []
for line in log.splitlines():
    if "libx264" in line:
        continue
    if any(k in line for k in ("pair: merged", "pushed=", "FAILED", "skipped", "align()", "orphan")):
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


def _ssh(host: str, cmd: str, timeout: int = 20) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
         "-o", "StrictHostKeyChecking=accept-new", host, cmd],
                       capture_output=True, text=True, timeout=timeout)
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
        for name, host in PIS:
            try:
                blob = json.loads(_ssh(host, PI_PROBE, timeout=12))
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
    for inv in (Path.home() / "umi-flow/provisioning/secrets",):
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
</style></head><body>
<h2 id="repo">…</h2><div id="cards">loading…</div>
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
