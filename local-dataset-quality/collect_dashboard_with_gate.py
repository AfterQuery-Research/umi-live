#!/usr/bin/env python3
"""Local-only collection dashboard with the continuous IK gate overlay."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "tools" / "collect_dashboard.py"
spec = importlib.util.spec_from_file_location("umi_collection_dashboard", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

GATE_STATUS = "/home/ubuntu/umi-flow-local/dataset-quality/results/status.json"
GATE_PROBE = f"cat {GATE_STATUS}"


def poll_gate() -> None:
    while True:
        try:
            gate = json.loads(base._ssh(base.SERVER, GATE_PROBE, timeout=12))
            with base.lock:
                base.state["gate"] = gate
                base.state["err"].pop("gate", None)
        except Exception as exc:
            with base.lock:
                base.state["err"]["gate"] = str(exc)[:80]
        time.sleep(2)


PAGE = r"""<!doctype html>
<html><head><title>UMI collection</title><meta charset="utf-8">
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--border:#30363d;--muted:#8b949e;--text:#e6edf3;--green:#39d353;--blue:#2f81f7;--white:#f0f6fc;--yellow:#d29922;--red:#f85149}
*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:28px;max-width:1500px}
h2{margin:0 0 16px}.cards{display:flex;flex-wrap:wrap;gap:10px}.card,.gate{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
.card{min-width:155px}.big{font-size:42px;font-weight:700;line-height:1.05}.lbl{color:var(--muted);font-size:13px;line-height:1.45}.ok{color:var(--green)}.warn{color:#e3b341}.bad{color:var(--red)}
.gate{margin-top:14px}.gate-head{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap}.gate-title{font-size:17px;font-weight:700}.gate-count{font-size:30px;font-weight:700}.gate-body{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;margin-top:14px}
.grid{display:grid;grid-template-rows:repeat(10,17px);grid-auto-flow:column;grid-auto-columns:17px;gap:4px;min-height:206px}.cell{width:17px;height:17px;border-radius:3px;border:1px solid rgba(240,246,252,.14)}
.standard{background:var(--green)}.optimized{background:var(--blue)}.fail{background:var(--white);border-color:#8b949e}.processing{background:var(--yellow)}.pending{background:#21262d}.error{background:var(--red)}
.legend{display:grid;grid-template-columns:13px auto;gap:7px 8px;align-items:center;font-size:12px;color:var(--muted);margin-bottom:13px}.swatch{width:13px;height:13px;border-radius:3px}
select{width:260px;max-width:100%;background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:7px;font:inherit;font-size:12px}
#events{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:15px;margin-top:14px;font-size:12px;line-height:1.7}
</style></head><body>
<h2 id="repo">connecting…</h2><div id="cards" class="cards"></div>
<section class="gate"><div class="gate-head"><div><div class="gate-title">Continuous optimization gate</div><div id="gateMeta" class="lbl">connecting…</div></div><div><span id="gatePassed" class="gate-count">—</span><span class="lbl"> passed</span></div></div>
<div class="gate-body"><div id="gateGrid" class="grid"></div><div><div class="legend"><span class="swatch standard"></span><span>standard pass</span><span class="swatch optimized"></span><span>pass with optimization</span><span class="swatch fail"></span><span>fail</span><span class="swatch processing"></span><span>processing</span><span class="swatch pending"></span><span>pending</span></div><select id="failedSelect"><option>Failed episodes (0)</option></select></div></div></section>
<div id="events"></div>
<script>
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function gateClass(ep){if(ep.state==='pass')return ep.selected_profile==='baseline_v3'?'standard':'optimized';return ep.state||'pending';}
function renderGate(g,b){
  const matching=!g.hub_repo||!b.repo||g.hub_repo===b.repo;
  if(!matching){document.getElementById('gateMeta').textContent=`waiting for ${b.repo}`;document.getElementById('gatePassed').textContent='—';document.getElementById('gateGrid').innerHTML='';return;}
  const standard=(g.episodes||[]).filter(e=>e.state==='pass'&&e.selected_profile==='baseline_v3').length;
  const optimized=(g.episodes||[]).filter(e=>e.state==='pass'&&e.selected_profile!=='baseline_v3').length;
  document.getElementById('gatePassed').textContent=g.passed??'—';
  document.getElementById('gateMeta').textContent=`${g.evaluated??0}/${g.dataset_total??0} evaluated · ${standard} standard · ${optimized} optimized · ${g.processing??0} processing · ${g.errors??0} errors · ${g.workers??'—'} CPU workers`;
  document.getElementById('gateGrid').innerHTML=(g.episodes||[]).map(ep=>{const reasons=(ep.failure_reasons||[]).join(', ');const title=`Episode ${ep.episode_index}: ${ep.state}${ep.selected_profile?' · '+ep.selected_profile:''}${reasons?' · '+reasons:''}`;return `<span class="cell ${gateClass(ep)}" title="${esc(title)}"></span>`;}).join('');
  const failed=(g.episodes||[]).filter(e=>e.state==='fail');
  document.getElementById('failedSelect').innerHTML=`<option>Failed episodes (${failed.length})</option>`+failed.map(ep=>`<option>Episode ${ep.episode_index}${ep.failure_reasons?.length?' — '+esc(ep.failure_reasons.join(', ')):''}</option>`).join('');
}
async function tick(){
  const s=await(await fetch('/status.json',{cache:'no-store'})).json(),b=s.box||{},hub=s.hub||{},g=s.gate||{};
  document.getElementById('repo').textContent=b.repo||'connecting to server…';
  const age=hub.last_push?Math.round(Date.now()/1000-hub.last_push):null,ageTxt=age===null?'—':age<90?age+'s ago':Math.round(age/60)+'m ago',justSent=age!==null&&age<90;
  let cards=`<div class=card><div class=big>${hub.episodes??'—'}</div><div class=lbl>episodes on the Hub</div><div class="lbl ${justSent?'ok':''}">last push ${ageTxt}${justSent?' ✓':''}</div></div>`;
  cards+=`<div class=card><div class=big>${hub.minutes??'—'}</div><div class=lbl>minutes on the Hub</div></div>`;
  cards+=`<div class=card><div class=big>${(b.processing||[]).length}</div><div class=lbl>processing now</div><div class=lbl>${(b.processing||[]).map(p=>esc(p.id.slice(-6)+' @ '+p.at+' ('+p.done+'/'+p.of+')')).join('<br>')||'idle'}</div></div>`;
  cards+=`<div class=card><div class=big>${(b.waiting_pair||[]).length}</div><div class=lbl>arms waiting for pair mate</div></div>`;
  for(const [name,pi] of Object.entries(s.pis||{})){cards+=`<div class=card><div class="big ${pi.pending>0?'warn':''}">${pi.pending}</div><div class=lbl>${esc(name)} uploads queued</div><div class="lbl ${pi.recorder==='active'?'ok':'bad'}">recorder ${esc(pi.recorder)} · ${pi.sent} sent</div></div>`;}
  const errs=Object.entries(s.err||{}).map(([k,v])=>`<span class=bad>${esc(k)}: ${esc(v)}</span>`).join(' · ');
  document.getElementById('cards').innerHTML=cards+(errs?`<div class=lbl>${errs}</div>`:'');
  renderGate(g,b);
  document.getElementById('events').innerHTML='<b>recent pipeline events</b><br>'+((b.events||[]).slice().reverse().map(esc).join('<br>')||'none');
}
tick();setInterval(tick,3000);
</script></body></html>"""


def main() -> None:
    base.PAGE = PAGE
    base.state.setdefault("gate", {})
    for function in (base.poll_box, base.poll_pis, base.poll_hub, poll_gate):
        threading.Thread(target=function, daemon=True).start()
    print(f"serving local gate dashboard http://localhost:{base.PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", base.PORT), base.Handler).serve_forever()


if __name__ == "__main__":
    main()
