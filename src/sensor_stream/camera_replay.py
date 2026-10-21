"""Standalone HTML replay comparing source and laptop receive timing."""

import json
import os
from pathlib import Path
from urllib.parse import quote

from .camera_inspection import image_path
from .inspection import load_capture


def write_replay(capture: Path, output: Path) -> int:
    _, events = load_capture(capture)
    frames = [e for e in events if e["type"] == "CameraSample"]
    if not frames:
        raise ValueError("Capture contains no camera frames")
    origin = frames[0]["received_monotonic_ns"]
    starts = {}
    rows = []
    for index, e in enumerate(frames):
        if e["file_check"] != "ok":
            raise ValueError(f"Frame {index} failed integrity check: {e['file_check']}")
        key = (e["session_id"], e["stream_id"])
        starts.setdefault(key, (e["source_time_ns"], e["received_monotonic_ns"]))
        source_start, receive_start = starts[key]
        receive = (e["received_monotonic_ns"] - origin) / 1e9
        # Align each stream clock at its first received frame. Later differences
        # show delivery lag growth without assuming the two clocks share an epoch.
        source = (receive_start - origin + e["source_time_ns"] - source_start) / 1e9
        rows.append(
            {
                "index": index,
                "source": source,
                "receive": receive,
                "url": quote(
                    os.path.relpath(
                        image_path(capture, e["image_path"]), output.resolve().parent
                    )
                ),
                "session": e["session_id"],
                "stream": e["stream_id"],
            }
        )
    payload = json.dumps(rows).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Camera timing replay</title>
<style>body{font:16px system-ui;background:#101722;color:#edf2fa;margin:24px}h1{font-size:24px}p{max-width:1000px;line-height:1.5}.views{display:flex;gap:20px}.view{flex:1;min-width:0}img{width:100%;height:55vh;object-fit:contain;background:#000}.controls{display:flex;gap:12px;align-items:center;margin:20px 0}input{flex:1}button,select{padding:8px}output{font-variant-numeric:tabular-nums}small{display:block;line-height:1.5} @media(max-width:700px){.views{display:block}img{height:30vh}}</style>
<h1>Camera timing replay</h1>
<p>Both views use the same saved frames. Source timing is aligned with the first received frame of each stream/session. The difference shows relative delivery lag, <b>not absolute camera latency</b>. Frames never received or decoded are absent. Keep this HTML and the capture images available on this computer.</p>
<div class="controls"><button id="play">Play</button><button id="reset">Reset</button><select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><input id="seek" aria-label="Replay time" type="range" min="0" step="0.01"><output id="time"></output></div>
<div class="views"><div class="view"><h2>Source-paced</h2><img id="source" alt="Source-paced recorded frame"><small id="sourceInfo"></small></div><div class="view"><h2>Receive-paced</h2><img id="receive" alt="Receive-paced recorded frame"><small id="receiveInfo"></small></div></div>
<p id="lag"></p><p>Display rotation: <select id="rotation"><option value="0">Original pixels</option><option value="90">90° clockwise</option><option value="180">180°</option><option value="270">90° counterclockwise</option></select> (view only; recorded files stay unchanged)</p>
<script>
const frames=FRAMES;
const seek=document.getElementById('seek'),time=document.getElementById('time'),play=document.getElementById('play');
const duration=Math.max(...frames.map(f=>Math.max(f.source,f.receive)));seek.max=duration;
let running=false,current=0,last=0;
const schedules={source:[...frames].sort((a,b)=>a.source-b.source),receive:[...frames].sort((a,b)=>a.receive-b.receive)};
function at(kind,t){const rows=schedules[kind];let lo=0,hi=rows.length;while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid][kind]<=t)lo=mid+1;else hi=mid;}return lo?rows[lo-1]:null;}
function render(){seek.value=current;time.textContent=current.toFixed(2)+' / '+duration.toFixed(2)+' s';for(const kind of ['source','receive']){const f=at(kind,current),img=document.getElementById(kind);if(f&&img.dataset.index!==String(f.index)){img.src=f.url;img.dataset.index=f.index;}document.getElementById(kind+'Info').textContent=f?'Frame '+f.index+' · source '+f.source.toFixed(2)+' s · receive '+f.receive.toFixed(2)+' s · session '+f.session.slice(0,8):'Waiting for frame';}const f=at('receive',current);document.getElementById('lag').textContent=f?'Relative delivery lag of displayed receive frame: '+(f.receive-f.source).toFixed(2)+' s':'No received frame yet';}
play.onclick=()=>{if(current>=duration)current=0;running=!running;play.textContent=running?'Pause':'Play';last=performance.now();render();};
document.getElementById('reset').onclick=()=>{current=0;running=false;play.textContent='Play';render();};seek.oninput=()=>{current=Number(seek.value);render();};
document.getElementById('rotation').onchange=e=>{for(const id of ['source','receive'])document.getElementById(id).style.transform='rotate('+e.target.value+'deg)';};
function tick(now){if(running){current=Math.min(duration,current+(now-last)/1000*Number(document.getElementById('speed').value));if(current>=duration){running=false;play.textContent='Play';}render();}last=now;requestAnimationFrame(tick);}render();requestAnimationFrame(tick);
</script></html>""".replace("FRAMES", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return len(rows)
