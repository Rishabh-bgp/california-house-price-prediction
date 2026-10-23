"use strict";
// In-browser inference for scikit-learn GradientBoostingRegressor models exported to JSON.
const INFO = {
  MedInc: ["Median household income ($10k)", 0.05], HouseAge: ["Median house age (years)", 1], AveRooms: ["Rooms per household", 0.05],
  AveBedrms: ["Bedrooms per household", 0.01], Population: ["Population", 10], AveOccup: ["People per household", 0.05],
  Latitude: ["Latitude", 0.01], Longitude: ["Longitude", 0.01],
};
let model, meta, map, pin;
const f32 = Math.fround, $ = id => document.getElementById(id);
function tree(t, x) { let n = 0; while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n]; return t.v[n]; }
function gbr(m, x) { let s = 0; for (const t of m.trees) s += tree(t, x); return m.base + m.lr * s; }
function predict(raw) {
  const x = model.features.map(f => f32(raw[f])), c = v => Math.min(Math.exp(v), model.cap);
  return {point: c(gbr(model.point, x)), lo: c(gbr(model.lo, x)), hi: c(gbr(model.hi, x))};
}
window.predictRaw = predict;
const usd = v => "$" + (Math.round(v * 1000) * 100).toLocaleString("en-US");
const colour = v => v < 1.5 ? "#1a9e9a" : v < 3 ? "#f4c27a" : v < 4.5 ? "#ff6f59" : "#8f1f1f";
function values() { const o = {}; model.features.forEach(f => o[f] = parseFloat($(f).value)); return o; }
function slider(f, box) {
  const [label, step] = INFO[f], d = document.createElement("div"); d.className = "field";
  d.innerHTML = `<label for="${f}">${label} <output id="o-${f}" for="${f}"></output></label><small>${f}</small>
    <input type="range" id="${f}" min="${meta.min[f]}" max="${meta.max[f]}" step="${step}" value="${meta.median[f]}">`;
  box.appendChild(d); d.querySelector("input").addEventListener("input", () => { render(); if (f === "Latitude" || f === "Longitude") movePin(); });
}
function movePin() { pin.setLatLng([+$("Latitude").value, +$("Longitude").value]); }
function render() {
  const r = values(), p = predict(r);
  model.features.forEach(f => { $("o-" + f).textContent = f === "MedInc" ? "$" + Math.round(r[f] * 10) + "k" : r[f]; });
  $("price").textContent = usd(p.point);
  $("capflag").hidden = p.point < model.cap * 0.97;
  $("range").textContent = `${usd(p.lo)} – ${usd(p.hi)}`;
  const sc = v => Math.min(100, Math.max(0, v / 5 * 100));
  $("band").style.left = sc(p.lo) + "%"; $("band").style.width = (sc(p.hi) - sc(p.lo)) + "%"; $("pin").style.left = sc(p.point) + "%";
  const below = meta.points.filter(q => q[2] < p.point).length;
  $("pct").textContent = `Higher than about ${Math.round(below / meta.points.length * 100)}% of California block groups (1990).`;
  drawHist(p.point);
}
function drawHist(v) {
  const c = $("hist"), x = c.getContext("2d"), W = c.width, H = c.height, b = meta.hist, mx = Math.max(...b), bw = (W - 20) / b.length;
  x.clearRect(0, 0, W, H);
  b.forEach((n, i) => { x.fillStyle = colour(0.1 + i * 0.2); const h = n / mx * (H - 30); x.fillRect(10 + i * bw + 1, H - 18 - h, bw - 2, h); });
  x.fillStyle = "#5b6573"; x.font = "12px Outfit, sans-serif";
  [["$10k", 0], ["$250k", 0.5], ["$500k cap", 1]].forEach(([t, k]) => x.fillText(t, 10 + k * (W - 70), H - 4));
  const px = 10 + (v - 0.1) / 5 * (W - 20); x.fillStyle = "#2b2d5c"; x.fillRect(px - 2, 4, 4, H - 22);
}
function setAll(vals) { model.features.forEach(f => { $(f).value = vals[f]; }); movePin(); render(); }
Promise.all([fetch("model.json").then(r => r.json()), fetch("data.json").then(r => r.json())]).then(([m, d]) => {
  model = m; meta = d;
  model.features.forEach(f => slider(f, $("sliders")));
  if (window.L) {
    map = L.map("map", {scrollWheelZoom: false}).setView([37.2, -119.5], 6);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
    const cv = L.canvas({padding: 0.3});
    meta.points.forEach(([la, lo, v]) => L.circleMarker([la, lo], {renderer: cv, radius: 3, weight: 0, fillOpacity: 0.75, fillColor: colour(v)}).addTo(map));
    pin = L.marker([meta.median.Latitude, meta.median.Longitude], {icon: L.divIcon({className: "", html: '<div class="pin-icon"></div>', iconSize: [18, 18]})}).addTo(map);
    map.on("click", e => {
      $("Latitude").value = e.latlng.lat.toFixed(2); $("Longitude").value = e.latlng.lng.toFixed(2); movePin(); render();
    });
  } else { pin = {setLatLng() {}}; }
  const pre = document.querySelector(".presets");
  Object.entries(meta.examples).forEach(([name, vals]) => {
    const b = document.createElement("button"); b.textContent = name; b.type = "button";
    b.addEventListener("click", () => { setAll(vals); if (map) map.setView([vals.Latitude, vals.Longitude], 9); }); pre.appendChild(b);
  });
  if (meta.coverage) $("cov").textContent = `(covered ${Math.round(meta.coverage * 100)}% of block groups in a hold-out test)`;
  setAll(meta.examples["Santa Monica"]); document.body.dataset.ready = "1";
});
