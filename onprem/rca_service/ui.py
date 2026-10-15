"""Single-page UI. No framework, no CDN, no build step: one string served by the same process.

The screen exists to show one thing. Their demo shows a traversal path and an agent pipeline;
this shows the residual, where a single bar stands over hundreds of flat ones. That picture is
the argument, so it is the centre of the page.
"""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deterministic Root Cause</title><style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e6e9ef;--dim:#8b93a7;
--hot:#ff5d4a;--cool:#3b82f6;--ok:#34d399;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;
align-items:baseline;gap:18px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}
.eq{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);font-size:12.5px}
.eq b{color:var(--hot);font-weight:600}
main{display:grid;grid-template-columns:260px 1fr 330px;gap:16px;padding:16px 24px;
align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.panel h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--dim);
margin:0 0 10px;font-weight:600}
.kv{display:flex;justify-content:space-between;gap:10px;padding:3px 0;font-size:13px}
.kv span:last-child{font-family:ui-monospace,Menlo,monospace;color:var(--ink)}
.dim{color:var(--dim)}
button{background:var(--cool);color:#fff;border:0;border-radius:7px;padding:9px 14px;
font-size:13px;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.12)}button:disabled{opacity:.5;cursor:default}
select{background:#0d1017;color:var(--ink);border:1px solid var(--line);border-radius:7px;
padding:8px 10px;font-size:13px;width:100%;margin-bottom:9px}
#chart{position:relative;height:340px;padding:6px 0 0}
#cv{width:100%;height:340px;display:block}
.legend{display:flex;gap:16px;font-size:12px;color:var(--dim);margin-top:6px;align-items:center}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px}
.cap{color:var(--dim);font-size:12px;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:4px 0;border-bottom:1px solid var(--line)}
td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
tr.hit td{color:var(--hot);font-weight:600}
pre{background:#0d1017;border:1px solid var(--line);border-radius:8px;padding:11px;
font-size:11.5px;overflow:auto;max-height:260px;color:#cbd3e1;margin:0}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.b-ok{background:rgba(52,211,153,.15);color:var(--ok)}
.b-no{background:rgba(255,93,74,.15);color:var(--hot)}
footer{padding:12px 24px;color:var(--dim);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:1100px){main{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>Deterministic Root Cause</h1>
  <div class="eq">s(t) = d(t) &minus; <b>M</b> d(t&minus;1) &nbsp;&middot;&nbsp; zero at every node except the origin</div>
  <div class="eq" style="margin-left:auto">0 learned parameters &middot; 0 labelled incidents</div>
</header>
<main>
  <div>
    <div class="panel"><h2>Operator</h2><div id="topo" class="dim">loading…</div></div>
    <div class="panel" style="margin-top:14px"><h2>Inject a fault</h2>
      <select id="kind"></select>
      <button id="go" style="width:100%">Inject &amp; locate</button>
      <div class="cap" id="timing"></div>
    </div>
  </div>
  <div class="panel">
    <h2>Residual, one bar per node</h2>
    <div id="chart"><canvas id="cv"></canvas></div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--hot)"></i>the origin: the operator cannot explain it</span>
      <span><i class="swatch" style="background:#2f4668"></i>every other node: flat under the operator</span>
    </div>
    <div class="cap" id="cap">Inject a fault. Every node is evaluated; the operator explains all
      of them but one.</div>
  </div>
  <div>
    <div class="panel"><h2>Finding</h2><div id="find" class="dim">—</div></div>
    <div class="panel" style="margin-top:14px"><h2>Healing intent (TMF921-shaped)</h2>
      <pre id="intent">—</pre></div>
  </div>
</main>
<footer>The explanation is the residual itself, not a traversal or an attention weight. Synthetic
counters on a topology of operator shape; see the repository for what this has and has not been
tested against.</footer>
<script>
const $=s=>document.querySelector(s);
async function boot(){
  const t=await (await fetch('v1/topology')).json();
  $('#topo').innerHTML=
    row('nodes',t.nodes)+row('edges',t.edges)+row('relations',t.relations.length)+
    row('stability',t.stability_margin)+row('build',t.operator_build_seconds+' s')+
    row('Riccati iters',t.riccati_iterations)+row('learned params',t.learned_parameters);
  const kinds=Object.entries(t.node_types).sort((a,b)=>a[1]-b[1]);
  $('#kind').innerHTML=kinds.map(([k,n])=>`<option value="${k}">${k} (${n} in topology)</option>`).join('');
}
const row=(k,v)=>`<div class="kv"><span class="dim">${k}</span><span>${v}</span></div>`;
function draw(res,originIdx,originName){
  const cv=$('#cv'), dpr=devicePixelRatio||1;
  const W=cv.clientWidth, H=340;
  cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const n=res.length, max=Math.max(...res,1e-9), pad=26, floor=H-22, top=18;
  const bw=Math.max(1,(W-2)/n);
  // median noise floor, so the eye has something to measure the spike against
  const sorted=[...res].sort((a,b)=>a-b), med=sorted[Math.floor(n/2)];
  const y=v=>floor-(v/max)*(floor-top);
  g.strokeStyle='#2a3142'; g.setLineDash([4,4]); g.beginPath();
  g.moveTo(0,y(med)); g.lineTo(W,y(med)); g.stroke(); g.setLineDash([]);
  g.fillStyle='#6b7488'; g.font='11px ui-monospace,Menlo,monospace';
  g.fillText('median '+med.toFixed(2),4,y(med)-5);
  for(let i=0;i<n;i++){
    const h=floor-y(res[i]);
    g.fillStyle = i===originIdx ? '#ff5d4a' : '#2f4668';
    g.fillRect(i*bw, y(res[i]), Math.max(1,bw-0.4), h);
  }
  // annotate the one bar that matters
  const ox=originIdx*bw, oy=y(res[originIdx]);
  g.strokeStyle='#ff5d4a'; g.lineWidth=1; g.beginPath();
  g.moveTo(ox+bw/2, oy-6); g.lineTo(ox+bw/2, 8); g.stroke();
  g.fillStyle='#ff5d4a'; g.font='600 12px ui-sans-serif,system-ui,sans-serif';
  const label=originName+'  '+res[originIdx].toFixed(2)+'  ('+(res[originIdx]/med).toFixed(1)+'x the median)';
  const tw=g.measureText(label).width;
  g.fillText(label, Math.min(Math.max(2,ox+bw/2-tw/2), W-tw-2), 6+8);
}
$('#go').onclick=async()=>{
  const b=$('#go'); b.disabled=true; b.textContent='running…';
  try{
    const r=await (await fetch('v1/demo?kind='+encodeURIComponent($('#kind').value))).json();
    draw(r.residual,r.origin_index,r.found);
    window._last=r;
    const hit=r.found===r.injected;
    $('#cap').innerHTML=`<b>${r.n_nodes-1}</b> nodes are flat under the operator. One is not.`;
    $('#find').innerHTML=
      `<div style="margin-bottom:8px"><span class="badge ${hit?'b-ok':'b-no'}">${hit?'CORRECT':'MISSED'}</span></div>`+
      row('injected',r.injected)+row('found',r.found)+row('confidence',r.confidence.toFixed(3))+
      row('separation z',r.separation_z.toFixed(1))+row('margin',r.margin.toFixed(3))+
      row('blast radius',r.blast_radius+' nodes downstream')+
      '<table style="margin-top:10px">'+r.top.map(x=>
        `<tr class="${x.node===r.found?'hit':''}"><td>${x.node}</td><td>${x.value.toFixed(3)}</td></tr>`).join('')+'</table>';
    $('#intent').textContent=JSON.stringify(r.intent,null,1);
    $('#timing').textContent=`${r.us_per_step} µs per step over ${r.steps} steps`;
  }catch(e){ $('#find').textContent='error: '+e; }
  b.disabled=false; b.textContent='Inject & locate';
};
boot();
addEventListener('resize',()=>{if(window._last)draw(window._last.residual,window._last.origin_index,window._last.found)});
</script></body></html>"""
