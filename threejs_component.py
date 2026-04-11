"""Three.js 3D BDH Walkthrough — self-contained HTML for st.components.v1.html()"""
 
def get_threejs_html(bdh_layer_activations, tf_layer_activations):
    import json, numpy as np
 
    N_NEURONS = 25   # 5×5 grid per layer
    def prep(stats_list, n=N_NEURONS):
        layers = []
        for s in stats_list:
            acts = s["activations"].mean(0)[:n]
            acts = (acts - acts.min()) / (acts.max() - acts.min() + 1e-8)
            layers.append(acts.tolist())
        return layers
 
    bdh_data  = json.dumps(prep(bdh_layer_activations))
    tf_data   = json.dumps(prep(tf_layer_activations))
    n_layers  = len(bdh_layer_activations)
 
    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  body{{margin:0;background:#0d1117;overflow:hidden;font-family:monospace}}
  canvas{{display:block}}
  #ui{{position:absolute;top:12px;left:50%;transform:translateX(-50%);
       display:flex;gap:10px;align-items:center;z-index:10}}
  .btn{{background:#f97316;color:#fff;border:none;padding:7px 16px;
        border-radius:8px;font-size:.82rem;font-weight:700;cursor:pointer}}
  .btn:hover{{background:#fb923c}}
  .btn.on{{background:#22c55e}}
  #lbdh{{position:absolute;top:52px;left:25%;transform:translateX(-50%);
          color:#f97316;font-weight:700;font-size:.95rem;pointer-events:none;
          text-shadow:0 0 12px #f97316}}
  #ltf{{position:absolute;top:52px;left:75%;transform:translateX(-50%);
         color:#3b82f6;font-weight:700;font-size:.95rem;pointer-events:none;
         text-shadow:0 0 12px #3b82f6}}
  #info{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
          color:#8b949e;font-size:.75rem;text-align:center;pointer-events:none}}
  #leg{{position:absolute;top:52px;right:14px;color:#8b949e;
         font-size:.73rem;line-height:1.9}}
  .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
</style>
</head>
<body>
<div id="ui">
  <button class="btn" id="bRot">⟳ Auto-rotate</button>
  <button class="btn" id="bPulse">⚡ Pulse Signal</button>
  <button class="btn" id="bReset">↺ Reset</button>
</div>
<div id="lbdh">🐉 BDH — Sparse ReLU</div>
<div id="ltf">🤖 Transformer — Dense GELU</div>
<div id="leg">
  <div><span class="dot" style="background:#f97316"></span>BDH active</div>
  <div><span class="dot" style="background:#4b6484;border:1px solid #5b7494"></span>BDH silent (~50%)</div>
  <div><span class="dot" style="background:#3b82f6"></span>Transformer (always on)</div>
  <div style="margin-top:5px;color:#6b7280">Drag · Scroll to zoom</div>
</div>
<div id="info">3D BDH Walkthrough — {n_layers} layers × {N_NEURONS} neurons (5×5) per side</div>
 
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const BDH_DATA  = {bdh_data};
const TF_DATA   = {tf_data};
const N_LAYERS  = {n_layers};
const N_NEURONS = {N_NEURONS};
const COLS      = 5;  // 5×5 grid
 
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 35, 95);
 
const W = window.innerWidth, H = window.innerHeight;
const camera = new THREE.PerspectiveCamera(52, W/H, 0.1, 200);
camera.position.set(0, 4, 30);
 
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W, H);
document.body.appendChild(renderer.domElement);
 
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const pl1 = new THREE.PointLight(0xf97316, 1.8, 70);
pl1.position.set(-9, 9, 14); scene.add(pl1);
const pl2 = new THREE.PointLight(0x3b82f6, 1.8, 70);
pl2.position.set(9, 9, 14);  scene.add(pl2);
 
// ── Pivot: ALL objects go in here so rotation works ────────────────────────
const pivot = new THREE.Group();
scene.add(pivot);
 
const SPHERE = new THREE.SphereGeometry(0.23, 10, 10);
const LAYER_SEP   = 4.8;
const SIDE_OFFSET = 8.5;
const GRID_SCALE  = 0.9;
 
function gridPos(idx) {{
  const col = idx % COLS, row = Math.floor(idx / COLS);
  const cx = (COLS-1)/2, cy = (Math.ceil(N_NEURONS/COLS)-1)/2;
  return {{x:(col-cx)*GRID_SCALE, y:(cy-row)*GRID_SCALE}};
}}
 
const bdhMeshes=[], tfMeshes=[], bdhMats=[], tfMats=[];
 
for (let l=0; l<N_LAYERS; l++) {{
  const z = -(l-(N_LAYERS-1)/2)*LAYER_SEP;
  for (let ni=0; ni<N_NEURONS; ni++) {{
    const gp = gridPos(ni);
 
    // ── BDH neuron ──────────────────────────────────────────────────────
    const bAct    = BDH_DATA[l][ni];
    const silent  = bAct < 0.08;   // ReLU hard zero threshold
    const bMat = new THREE.MeshPhongMaterial({{
      color:    silent ? 0x4b6484 : 0xf97316,
      emissive: silent ? new THREE.Color(0x1a2535) : new THREE.Color(0xf97316).multiplyScalar(bAct*0.55),
      transparent: true,
      opacity:  silent ? 0.55 : 0.88 + bAct*0.12,
      shininess: silent ? 10 : 60,
    }});
    const bMesh = new THREE.Mesh(SPHERE, bMat);
    bMesh.position.set(-SIDE_OFFSET+gp.x, gp.y, z);
    bMesh.scale.setScalar(silent ? 0.58 : 0.78+bAct*0.48);
    bMesh.userData = {{l, ni, act:bAct, silent}};
    pivot.add(bMesh);   // ← pivot, not scene
    bdhMeshes.push(bMesh); bdhMats.push(bMat);
 
    // ── Transformer neuron (GELU: always active) ────────────────────────
    const tAct = TF_DATA[l][ni];
    const tMat = new THREE.MeshPhongMaterial({{
      color:   0x3b82f6,
      emissive:new THREE.Color(0x3b82f6).multiplyScalar(0.38+tAct*0.38),
      transparent:true, opacity:0.80+tAct*0.18, shininess:50,
    }});
    const tMesh = new THREE.Mesh(SPHERE, tMat);
    tMesh.position.set(SIDE_OFFSET+gp.x, gp.y, z);
    tMesh.scale.setScalar(0.72+tAct*0.42);
    tMesh.userData = {{l, ni, act:tAct, silent:false}};
    pivot.add(tMesh);   // ← pivot
    tfMeshes.push(tMesh); tfMats.push(tMat);
  }}
}}
 
// ── Edges — also added to pivot so they rotate with neurons ───────────────
const bdhEdgeMats = [], tfEdgeMats = [];
 
function addEdges(meshes, color, maxEdgesPerLayer, edgeMatsArr) {{
  const pts = []; 
  for (let l=0; l<N_LAYERS-1; l++) {{
    const base = l*N_NEURONS;
    let count = 0;
    // Connect neurons between current layer (l) and next layer (l+1)
    for (let ni=0; ni<N_NEURONS; ni++) {{
      const src = meshes[base+ni];
      if (src.userData.silent) continue;
      
      for (let nj=0; nj<N_NEURONS; nj+=2) {{ // nj+=2 spreads edges out visually
        const dst = meshes[base+N_NEURONS+nj];
        if (dst.userData.silent) continue;
        
        if (count < maxEdgesPerLayer) {{
          pts.push(src.position.x, src.position.y, src.position.z,
                   dst.position.x, dst.position.y, dst.position.z);
          count++;
        }}
      }}
    }}
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.LineBasicMaterial({{color, transparent:true, opacity:0.18}});
  const lines = new THREE.LineSegments(geo, mat);
  pivot.add(lines);
  edgeMatsArr.push(mat);
  return mat;
}}

// Update the function calls right below it to pass edges PER LAYER:
addEdges(bdhMeshes, 0xf97316, 40, bdhEdgeMats); // 40 lines max between each BDH layer
addEdges(tfMeshes,  0x3b82f6, 80, tfEdgeMats);  // 80 lines max between each TF layer
 
// ── Layer separator planes ─────────────────────────────────────────────────
for (let l=0; l<N_LAYERS; l++) {{
  const z = -(l-(N_LAYERS-1)/2)*LAYER_SEP;
  const pl = new THREE.Mesh(
    new THREE.PlaneGeometry(14, 0.015),
    new THREE.MeshBasicMaterial({{color:0x374151, transparent:true, opacity:0.4}})
  );
  pl.position.set(0,-2.4,z);
  pivot.add(pl);  // ← pivot
}}
 
// ── Mouse orbit ─────────────────────────────────────────────────────────────
let drag=false, pm={{x:0,y:0}}, rotX=0.1, rotY=0;
renderer.domElement.addEventListener("mousedown", e=>{{drag=true; pm={{x:e.clientX,y:e.clientY}}}});
renderer.domElement.addEventListener("mouseup",   ()=>drag=false);
renderer.domElement.addEventListener("mousemove", e=>{{
  if(!drag) return;
  rotY+=(e.clientX-pm.x)*0.008; rotX+=(e.clientY-pm.y)*0.008;
  pm={{x:e.clientX,y:e.clientY}};
}});
renderer.domElement.addEventListener("wheel", e=>{{
  camera.position.z=Math.max(8,Math.min(55,camera.position.z+e.deltaY*0.04));
}});
 
// ── State ─────────────────────────────────────────────────────────────────
let autoRot=false, pulseActive=false, pulseT=0;
const clock=new THREE.Clock();
 
document.getElementById("bRot").onclick=function(){{
  autoRot=!autoRot;
  this.textContent=autoRot?"⏸ Pause":"⟳ Auto-rotate";
  this.classList.toggle("on",autoRot);
}};
document.getElementById("bPulse").onclick=function(){{
  pulseActive=true; pulseT=0;
  this.textContent="⚡ Pulsing...";
  setTimeout(()=>{{pulseActive=false; this.textContent="⚡ Pulse Signal";}},3000);
}};
document.getElementById("bReset").onclick=()=>{{
  rotX=0.1; rotY=0; camera.position.set(0,4,30);
}};
 
// ── Animate ───────────────────────────────────────────────────────────────
function animate(){{
  requestAnimationFrame(animate);
  const t=clock.getElapsedTime();
  if(autoRot) rotY+=0.005;
  pivot.rotation.y=rotY; pivot.rotation.x=rotX;
 
  // Idle pulse on active BDH neurons
  for(let i=0;i<bdhMeshes.length;i++){{
    if(!bdhMeshes[i].userData.silent){{
      const p=1+0.07*Math.sin(t*2.4+i*0.35);
      bdhMeshes[i].scale.setScalar((0.78+bdhMeshes[i].userData.act*0.48)*p);
    }}
  }}
 
  // Signal pulse sweep — lights up neurons AND edges
  if(pulseActive){{
    pulseT+=0.018;
    const waveZ=(pulseT*3.2-(N_LAYERS)*0.5)*LAYER_SEP;
    // Neurons
    for(let i=0;i<bdhMeshes.length;i++){{
      const dist=Math.abs(bdhMeshes[i].position.z-waveZ);
      const glow=Math.max(0,1-dist/2.2);
      if(!bdhMeshes[i].userData.silent){{
        bdhMats[i].emissive=new THREE.Color(1,0.6,0.1).multiplyScalar(glow*0.9+bdhMeshes[i].userData.act*0.35);
        bdhMats[i].color=new THREE.Color(0xffffff).lerp(new THREE.Color(0xf97316),1-glow*0.6);
      }} else {{
        // Silent neurons flash dim on pulse wavefront
        bdhMats[i].opacity=0.55+glow*0.35;
        bdhMats[i].emissive=new THREE.Color(0x2a3d55).multiplyScalar(glow*0.5);
      }}
    }}
    // Edges brighten on pulse
    for(let m of bdhEdgeMats) m.opacity=0.18+Math.sin(pulseT*8)*0.25;
    for(let m of tfEdgeMats)  m.opacity=0.18;
  }} else {{
    // Reset to normal
    for(let i=0;i<bdhMeshes.length;i++){{
      const act=bdhMeshes[i].userData.act;
      const sl=bdhMeshes[i].userData.silent;
      bdhMats[i].color.set(sl?0x4b6484:0xf97316);
      bdhMats[i].emissive=sl?new THREE.Color(0x1a2535):new THREE.Color(0xf97316).multiplyScalar(act*0.55);
      bdhMats[i].opacity=sl?0.55:0.88+act*0.12;
    }}
    for(let m of bdhEdgeMats) m.opacity=0.18;
    for(let m of tfEdgeMats)  m.opacity=0.18;
  }}
 
  renderer.render(scene,camera);
}}
animate();
 
window.addEventListener("resize",()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});
</script>
</body></html>"""
 
