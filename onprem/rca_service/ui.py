"""Single-page UI. No framework, no CDN, no build step: one string served by the same process.

The screen has to carry an argument, not just a number. The argument has three beats and they
are the same three panels as the figure in the paper: the fault hiding in raw counters, the
field spreading across the blast radius, and the operator leaving exactly one node unexplained.
Everything else on the page is in service of those three.
"""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deterministic Root Cause</title><style>
:root{--bg:#0e1014;--panel:#161920;--line:#242935;--ink:#e8ebf2;--dim:#8b93a7;--dim2:#616b80;
--hot:#ff5d4a;--cool:#4b8bf5;--ok:#34d399;--warn:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:16px 26px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;
gap:20px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650}
.eq{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);font-size:12.5px}
.eq b{color:var(--hot)}
.pill{margin-left:auto;font-size:12px;color:var(--dim)}
.pill b{color:var(--ok)}
main{padding:18px 26px 40px;max-width:1500px;margin:0 auto}
.ctl{display:flex;gap:10px;align-items:center;margin-bottom:18px;flex-wrap:wrap}
select{background:#0c0f15;color:var(--ink);border:1px solid var(--line);border-radius:7px;
padding:9px 11px;font-size:13px}
button{background:var(--cool);color:#fff;border:0;border-radius:7px;padding:10px 18px;
font-size:13px;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.12)}button:disabled{opacity:.45;cursor:default}
.note{color:var(--dim2);font-size:12.5px}
.act{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 18px;
margin-bottom:14px}
.act h3{margin:0 0 2px;font-size:14px;font-weight:650}
.act h3 em{font-style:normal;color:var(--dim);font-weight:500}
.act p{margin:0 0 10px;color:var(--dim);font-size:12.5px}
.hm{width:100%;height:150px;display:block;border-radius:6px;background:#0a0c10;
image-rendering:pixelated}
.axis{display:flex;justify-content:space-between;color:var(--dim2);font-size:11px;margin-top:5px;
font-family:ui-monospace,Menlo,monospace}
.grid{display:grid;grid-template-columns:1.25fr .85fr;gap:14px;align-items:start}
.kv{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;
border-bottom:1px solid var(--line)}
.kv span:last-child{font-family:ui-monospace,Menlo,monospace}
.big{font-size:27px;font-weight:700;font-family:ui-monospace,Menlo,monospace;line-height:1.15}
.big.hot{color:var(--hot)}.big.ok{color:var(--ok)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:4px 0 2px}
.card{background:#0c0f15;border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.card .lab{color:var(--dim2);font-size:11px;text-transform:uppercase;letter-spacing:.7px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin:0 0 10px}
pre{background:#0a0c10;border:1px solid var(--line);border-radius:8px;padding:11px;font-size:11px;
overflow:auto;max-height:300px;color:#c6cedd;margin:0}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:4px 0;border-bottom:1px solid var(--line)}
td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
tr.hit td{color:var(--hot);font-weight:650}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:650}
.b-ok{background:rgba(52,211,153,.14);color:var(--ok)}.b-no{background:rgba(255,93,74,.14);color:var(--hot)}
footer{padding:14px 26px;color:var(--dim2);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:1000px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>Deterministic Root Cause</h1>
  <div class="eq">s(t) = d(t) &minus; <b>M</b> d(t&minus;1)</div>
  <div class="pill"><b>0</b> learned parameters &nbsp;&middot;&nbsp; <b>0</b> labelled incidents
    &nbsp;&middot;&nbsp; the model it matches has 241,222</div>
</header>
<main>
  <div class="ctl">
    <span class="note">Start a fault somewhere in the network:</span>
    <select id="kind"></select>
    <button id="go">Run the incident</button>
    <span class="note" id="status"></span>
  </div>

  <div class="act">
    <h3>1 &nbsp; The fault is hiding <em>&mdash; raw counters, every node, over time</em></h3>
    <p id="p1">A degradation begins and propagates. Nothing here breaches a threshold for a long
      while, which is exactly why these are expensive.</p>
    <canvas id="c1" class="hm"></canvas><div class="axis"><span>time &rarr;</span><span id="a1"></span></div>
  </div>

  <div class="act">
    <h3>2 &nbsp; The field says many nodes are sick <em>&mdash; estimated degradation</em></h3>
    <p id="p2">Filtering the counters through the known dynamics recovers the degradation field.
      Now the problem is visible, and it is everywhere downstream of the cause.</p>
    <canvas id="c2" class="hm"></canvas><div class="axis"><span>time &rarr;</span><span id="a2"></span></div>
  </div>

  <div class="act">
    <h3>3 &nbsp; The operator explains all but one <em>&mdash; the residual</em></h3>
    <p id="p3">Apply the operator and subtract. Every node that is merely suffering the
      consequences is accounted for and goes dark. What remains is the cause.</p>
    <canvas id="c3" class="hm"></canvas><div class="axis"><span>time &rarr;</span><span id="a3"></span></div>
  </div>

  <div class="cards" id="cards"></div>

  <div class="grid" style="margin-top:14px">
    <div class="act"><h2>Final residual, one bar per node</h2>
      <canvas id="cv" style="width:100%;height:230px;display:block"></canvas>
      <div class="note" id="cap" style="margin-top:8px">Run an incident.</div>
    </div>
    <div>
      <div class="act"><h2>Finding</h2><div id="find" class="note">&mdash;</div></div>
      <div class="act"><h2>Healing intent (TMF921-shaped)</h2><pre id="intent">&mdash;</pre></div>
    </div>
  </div>
</main>
<footer>The explanation is the residual itself, not a traversal path or an attention weight.
Synthetic counters on a topology of operator shape. This method has been run against real
telemetry once, on a stack carrying no traffic where its precondition provably fails; see the
repository for what it has and has not been tested against.</footer>
<script>
const $=s=>document.querySelector(s);
const row=(k,v)=>`<div class="kv"><span class="note">${k}</span><span>${v}</span></div>`;
let S=null;

function heat(id,b64,steps,n,hot){
  const cv=$(id), dpr=devicePixelRatio||1, W=cv.clientWidth, H=150;
  cv.width=steps; cv.height=n; cv.style.height=H+'px';
  const g=cv.getContext('2d'), img=g.createImageData(steps,n);
  const bin=atob(b64);
  for(let y=0;y<n;y++)for(let x=0;x<steps;x++){
    const v=bin.charCodeAt(y*steps+x)|0, o=(y*steps+x)*4;
    if(hot){ img.data[o]=Math.min(255,40+v); img.data[o+1]=Math.round(v*0.32);
             img.data[o+2]=Math.round(v*0.26); }
    else   { img.data[o]=Math.round(v*0.30); img.data[o+1]=Math.round(v*0.52);
             img.data[o+2]=Math.min(255,60+v); }
    img.data[o+3]=255;
  }
  g.putImageData(img,0,0);
}

function bars(res,oi,name){
  const cv=$('#cv'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=230;
  cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');
  g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,W,H);
  const n=res.length,max=Math.max(...res,1e-9),floor=H-20,top=26,bw=Math.max(1,(W-2)/n);
  const med=[...res].sort((a,b)=>a-b)[Math.floor(n/2)],y=v=>floor-(v/max)*(floor-top);
  g.strokeStyle='#2a3142';g.setLineDash([4,4]);g.beginPath();
  g.moveTo(0,y(med));g.lineTo(W,y(med));g.stroke();g.setLineDash([]);
  g.fillStyle='#616b80';g.font='11px ui-monospace,Menlo,monospace';
  g.fillText('median '+med.toFixed(2),4,y(med)-5);
  for(let i=0;i<n;i++){g.fillStyle=i===oi?'#ff5d4a':'#2b3f5e';
    g.fillRect(i*bw,y(res[i]),Math.max(1,bw-0.4),floor-y(res[i]));}
  const ox=oi*bw+bw/2;
  g.strokeStyle='#ff5d4a';g.beginPath();g.moveTo(ox,y(res[oi])-6);g.lineTo(ox,14);g.stroke();
  g.fillStyle='#ff5d4a';g.font='600 12px ui-sans-serif,system-ui,sans-serif';
  const t=name+'  '+res[oi].toFixed(2)+'  = '+(res[oi]/med).toFixed(0)+'x the median';
  const w=g.measureText(t).width;
  g.fillText(t,Math.min(Math.max(2,ox-w/2),W-w-2),12);
}

async function boot(){
  const t=await (await fetch('v1/topology')).json();
  $('#kind').innerHTML=Object.entries(t.node_types).sort((a,b)=>a[1]-b[1])
    .map(([k,n])=>`<option value="${k}">${k} (${n} in topology)</option>`).join('');
}

$('#go').onclick=async()=>{
  const b=$('#go');b.disabled=true;b.textContent='running…';$('#status').textContent='';
  try{
    const r=await (await fetch('v1/story?kind='+encodeURIComponent($('#kind').value))).json();
    S=r;
    heat('#c1',r.panels.counters,r.steps,r.n_nodes,false);
    heat('#c2',r.panels.field,r.steps,r.n_nodes,false);
    heat('#c3',r.panels.residual,r.steps,r.n_nodes,true);
    for(const i of [1,2,3]) $('#a'+i).textContent=r.steps+' steps  ·  '+r.n_nodes+' nodes';
    $('#p1').innerHTML=`A <b>${r.origin_type}</b> fault starts at step <b>${r.onset}</b> and ramps
      over ${r.ramp} steps. A static threshold on these counters does not fire until step
      <b>${r.threshold_alarm_step??'never'}</b>.`;
    $('#p2').innerHTML=`The field spreads to <b>${r.blast_radius}</b> nodes downstream. Every one
      of them looks broken. Only one of them is.`;
    $('#p3').innerHTML=`The operator resolves the origin at step <b>${r.operator_resolved_step}</b>
      — <b>${r.lead_steps}</b> steps before the threshold would have alarmed. One row stays lit.`;
    const hit=r.injected===r.found;
    $('#cards').innerHTML=
      card('lead over a static alarm',`${r.lead_steps} steps`,'ok','detected at '+r.operator_resolved_step+', alarm at '+r.threshold_alarm_step)+
      card('blast radius',`${r.blast_radius} nodes`,'','all explained but one; revenue at risk '+r.revenue_at_risk)+
      card('cost to answer',`${r.us_per_step} µs`,'','per step, 0 parameters, deterministic');
    bars(r.final_residual,r.origin_index,r.found);
    $('#cap').innerHTML=`<b>${r.n_nodes-1}</b> nodes are flat under the operator. One is not.`;
    $('#find').innerHTML=`<div style="margin-bottom:8px"><span class="badge ${hit?'b-ok':'b-no'}">${hit?'CORRECT':'MISSED'}</span></div>`+
      row('injected',r.injected)+row('found',r.found)+row('domain',r.domain)+
      row('confidence',r.confidence.toFixed(3))+row('separation z',r.separation_z.toFixed(1))+
      '<table style="margin-top:10px">'+r.top.map(x=>
        `<tr class="${x.node===r.found?'hit':''}"><td>${x.node}</td><td>${x.value.toFixed(3)}</td></tr>`).join('')+'</table>';
    $('#intent').textContent=JSON.stringify(r.intent,null,1);
    $('#status').textContent=`resolved in ${r.us_per_step} µs per step`;
  }catch(e){$('#status').textContent='error: '+e;}
  b.disabled=false;b.textContent='Run the incident';
};
const card=(lab,val,cls,sub)=>`<div class="card"><div class="lab">${lab}</div>
  <div class="big ${cls}">${val}</div><div class="note" style="font-size:11.5px">${sub}</div></div>`;
addEventListener('resize',()=>{if(S)bars(S.final_residual,S.origin_index,S.found)});
boot();
</script></body></html>"""
