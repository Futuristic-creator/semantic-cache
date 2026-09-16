import json

with open("results.json") as f:
    RESULTS = json.load(f)

DATA_JSON = json.dumps(RESULTS)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Semantic Cache Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<script>
window.chartJsReady = new Promise((resolve, reject) => {
  if (window.Chart) { resolve(); return; }
  const fallbacks = [
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js",
    "https://unpkg.com/chart.js@4.4.4/dist/chart.umd.min.js"
  ];
  function tryNext(i) {
    if (i >= fallbacks.length) { reject(new Error("all Chart.js sources failed")); return; }
    const s = document.createElement('script');
    s.src = fallbacks[i];
    s.onload = () => resolve();
    s.onerror = () => tryNext(i + 1);
    document.head.appendChild(s);
  }
  tryNext(0);
});
</script>
<style>
  :root{
    --bg:#14171C;
    --panel:#1B1F26;
    --panel-2:#20242C;
    --border:#2A2F38;
    --text:#E7E9EC;
    --muted:#8B9198;
    --signal:#3FB88F;
    --noise:#E3A83B;
    --line-naive:#5C8AC9;
    --line-gptcache:#C97BD8;
    --line-redis:#E3A83B;
    --line-adaptive:#3FB88F;
    --line-confidence_tiered:#F2545B;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:'Space Grotesk', sans-serif;
    line-height:1.55;
  }
  .mono{ font-family:'IBM Plex Mono', monospace; }
  .wrap{ max-width:980px; margin:0 auto; padding:64px 24px 96px; }
  header.hero{ border-bottom:1px solid var(--border); padding-bottom:40px; margin-bottom:40px; }
  .kicker{ color:var(--muted); font-size:15px; margin:0 0 18px; max-width:640px; }
  .hero-stat{
    font-family:'IBM Plex Mono', monospace;
    font-weight:600;
    font-size:clamp(56px, 9vw, 104px);
    line-height:1;
    color:var(--noise);
    margin:0 0 4px;
  }
  .hero-stat sup{ font-size:0.35em; color:var(--muted); font-weight:500; top:-0.6em; }
  .hero-label{ font-size:19px; max-width:600px; margin:0 0 26px; }
  .hero-sub{ color:var(--muted); font-size:15px; max-width:640px; }
  .hero-sub b{ color:var(--signal); font-weight:600; }
  h2{
    font-size:22px; font-weight:600; margin:0 0 6px;
    letter-spacing:-0.01em;
  }
  .section-desc{ color:var(--muted); font-size:14.5px; max-width:680px; margin:0 0 24px; }
  section{ margin-bottom:56px; }
  .panel{
    background:var(--panel); border:1px solid var(--border);
    border-radius:6px; padding:24px;
  }
  .chart-row{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  @media (max-width:760px){ .chart-row{ grid-template-columns:1fr; } }
  .chart-box{ position:relative; height:280px; }
  .legend{ display:flex; flex-wrap:wrap; gap:16px; margin-top:16px; font-size:13px; color:var(--muted); }
  .legend span{ display:inline-flex; align-items:center; gap:7px; }
  .dot{ width:9px; height:9px; border-radius:2px; display:inline-block; }
  table{ width:100%; border-collapse:collapse; font-size:14px; }
  th, td{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); }
  th{ color:var(--muted); font-weight:500; font-size:12.5px; }
  td.num, th.num{ font-family:'IBM Plex Mono', monospace; text-align:right; }
  .trap-cell{ color:var(--noise); font-weight:600; }
  .ok-cell{ color:var(--signal); font-weight:600; }
  .matrix-table td, .matrix-table th{ font-size:13.5px; vertical-align:top; }
  .matrix-table td:first-child{ color:var(--text); font-weight:500; white-space:nowrap; }
  footer{ border-top:1px solid var(--border); padding-top:24px; color:var(--muted); font-size:13px; }
  footer p{ max-width:680px; }
  a{ color:var(--line-naive); }
</style>
</head>
<body>
<div class="wrap">

  <header class="hero">
    <p class="kicker">Semantic caching benchmark — comparing caching <em>policies</em> (not just libraries) on cost, latency, and a failure mode most benchmarks skip: serving the wrong cached answer.</p>
    <p class="hero-stat">20<span class="mono" style="color:var(--muted); font-size:0.5em;"> / 20</span></p>
    <p class="hero-label">trap queries fooled the standard similarity-threshold baselines at a typical operating threshold.</p>
    <p class="hero-sub">A "trap" is a query worded almost identically to a cached one but asking something different — e.g. <em>"cancel my subscription"</em> vs <em>"cancel my subscription renewal reminder."</em> The naive, GPTCache-style, and Redis-style backends each served a wrong cached answer on effectively every trap in this set. The adaptive per-entry-threshold backend caught <b>6 of 20</b> — a real improvement, not a solved problem.</p>
  </header>

  <section>
    <h2>Hit rate vs. similarity threshold</h2>
    <p class="section-desc">Higher isn't automatically better — a higher hit rate paired with falling precision (right chart) means more requests are being answered from cache, but a growing share of those answers are wrong.</p>
    <div class="panel chart-row">
      <div class="chart-box"><canvas id="hitRateChart"></canvas></div>
      <div class="chart-box"><canvas id="precisionChart"></canvas></div>
    </div>
    <div class="legend" id="sharedLegend"></div>
  </section>

  <section>
    <h2>Cost saved vs. no caching, at each threshold</h2>
    <p class="section-desc">Modeled against illustrative per-call cost assumptions (see README) — the shape of the curve matters more than the absolute dollar figure.</p>
    <div class="panel">
      <div class="chart-box" style="height:260px;"><canvas id="costChart"></canvas></div>
    </div>
  </section>

  <section>
    <h2>Where each backend actually fails</h2>
    <p class="section-desc">Category breakdown at threshold = 0.25. "Trap" is the column that matters: a false hit there means the system confidently served the wrong answer.</p>
    <div class="panel">
      <table id="categoryTable"></table>
    </div>
  </section>

  <section>
    <h2>Try it yourself</h2>
    <p class="section-desc">Type a query and see how each backend responds live, at your chosen threshold. Try a deliberate trap: something worded like a cached topic but asking something different. A "HIT" just means a backend decided to serve a cached answer &mdash; judge for yourself whether the matched topic actually answers what you asked.</p>
    <div class="panel">
      <div id="liveTopics" class="mono" style="color:var(--muted); font-size:12.5px; margin-bottom:16px; line-height:1.7;"></div>
      <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px;">
        <input id="liveQuery" type="text" placeholder="e.g. how do I cancel my subscription renewal reminder"
               style="flex:1; min-width:260px; background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:10px 12px; color:var(--text); font-family:'Space Grotesk',sans-serif; font-size:14px;">
        <input id="liveThreshold" type="number" step="0.05" min="0" max="1" value="0.25"
               style="width:90px; background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:10px 12px; color:var(--text); font-family:'IBM Plex Mono',monospace; font-size:14px;">
        <button id="liveRunBtn"
                style="background:var(--signal); border:none; border-radius:4px; padding:10px 18px; color:#0F1512; font-weight:600; cursor:pointer; font-family:'Space Grotesk',sans-serif; font-size:14px;">
          Run query
        </button>
      </div>
      <div id="liveResults"></div>
      <p id="liveNote" class="section-desc" style="margin-top:14px; margin-bottom:0;">Requires the Flask demo server (<span class="mono">python3 app.py</span>) — this panel calls a live API, not precomputed data.</p>
    </div>
  </section>

  <section>
    <h2>Cost savings calculator</h2>
    <p class="section-desc">Plug in your own volume and per-call cost. Threshold snaps to the values actually measured in this benchmark &mdash; no extrapolation.</p>
    <div class="panel">
      <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:20px;">
        <div>
          <label style="display:block; font-size:12.5px; color:var(--muted); margin-bottom:6px;">Queries per day</label>
          <input id="calcVolume" type="number" value="100000" step="1000" min="0"
                 style="width:140px; background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:8px 10px; color:var(--text); font-family:'IBM Plex Mono',monospace;">
        </div>
        <div>
          <label style="display:block; font-size:12.5px; color:var(--muted); margin-bottom:6px;">Cost per LLM call (USD)</label>
          <input id="calcCost" type="number" value="0.0015" step="0.0001" min="0"
                 style="width:140px; background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:8px 10px; color:var(--text); font-family:'IBM Plex Mono',monospace;">
        </div>
        <div>
          <label style="display:block; font-size:12.5px; color:var(--muted); margin-bottom:6px;">Similarity threshold</label>
          <select id="calcThreshold" style="width:140px; background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:8px 10px; color:var(--text); font-family:'IBM Plex Mono',monospace;"></select>
        </div>
      </div>
      <table id="calcTable"></table>
      <p style="font-size:12px; color:var(--muted); margin:14px 0 0;">Dollar figures scale the measured % cost saved by your inputs &mdash; not a separately measured number. Threshold list is exactly what this benchmark tested; nothing in between is interpolated.</p>
    </div>
  </section>

  <section>
    <h2>Decision matrix</h2>
    <p class="section-desc">What this benchmark suggests for real decisions, not just which bar is tallest.</p>
    <div class="panel">
      <table class="matrix-table">
        <tr><th>Situation</th><th>Recommendation</th><th>Why</th></tr>
        <tr><td>Prototyping, low volume</td><td>Naive baseline</td><td>Simplicity wins before eviction/precision issues appear at scale</td></tr>
        <tr><td>High volume, cost-sensitive, error-tolerant</td><td>GPTCache-style (LRU)</td><td>Bounded memory; accept the precision tradeoff explicitly and monitor false-hit rate</td></tr>
        <tr><td>Compliance-sensitive (fintech, health)</td><td>Add a verification layer above any policy</td><td>A false hit here is a correctness incident, not an inconvenience</td></tr>
        <tr><td>Long-running system with feedback data</td><td>Adaptive per-entry threshold</td><td>Justifies the calibration cost; less manual tuning over time</td></tr>
        <tr><td>Any customer-facing high-stakes flow</td><td>Don't rely on similarity threshold alone</td><td>Every policy tested here has a real false-hit floor</td></tr>
      </table>
    </div>
  </section>

  <footer>
    <p>Methodology, dataset, cost-model assumptions, and how to swap in real SDKs (GPTCache / Redis / a hosted embedding model) are documented in <span class="mono">README.md</span>. Backends here are faithful reimplementations of each product's documented default policy, run on a shared TF-IDF embedder — built this way because this environment has no internet access to install the real SDKs or a pretrained embedding model.</p>
  </footer>

</div>

<script>
const RESULTS = __DATA_JSON__;

const BACKEND_ORDER = ["naive", "gptcache", "redis", "adaptive", "confidence_tiered"];
const COLORS = {
  naive: getComputedStyle(document.documentElement).getPropertyValue('--line-naive').trim(),
  gptcache: getComputedStyle(document.documentElement).getPropertyValue('--line-gptcache').trim(),
  redis: getComputedStyle(document.documentElement).getPropertyValue('--line-redis').trim(),
  adaptive: getComputedStyle(document.documentElement).getPropertyValue('--line-adaptive').trim(),
  confidence_tiered: getComputedStyle(document.documentElement).getPropertyValue('--line-confidence_tiered').trim(),
};

const thresholds = RESULTS.meta.threshold_sweep;

function seriesFor(key, field){
  return RESULTS.backends[key].by_threshold.map(r => r[field]);
}

const gridColor = "#2A2F38";
const tickColor = "#8B9198";
const fontFamily = "'IBM Plex Mono', monospace";

function baseOptions(yLabel, yMax){
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{ display:false } },
    scales:{
      x:{ grid:{ color:gridColor }, ticks:{ color:tickColor, font:{ family:fontFamily, size:11 } },
          title:{ display:true, text:'similarity threshold', color:tickColor, font:{ family:fontFamily, size:11 } } },
      y:{ grid:{ color:gridColor }, ticks:{ color:tickColor, font:{ family:fontFamily, size:11 } },
          min:0, max:yMax,
          title:{ display:true, text:yLabel, color:tickColor, font:{ family:fontFamily, size:11 } } }
    }
  };
}

function makeLineDatasets(field){
  return BACKEND_ORDER.map(key => ({
    label: RESULTS.backends[key].name,
    data: seriesFor(key, field),
    borderColor: COLORS[key],
    backgroundColor: COLORS[key],
    tension: 0.25,
    pointRadius: 3,
    borderWidth: 2,
  }));
}

window.chartJsReady.then(() => {
  new Chart(document.getElementById('hitRateChart'), {
    type:'line',
    data:{ labels: thresholds, datasets: makeLineDatasets('hit_rate') },
    options: baseOptions('hit rate', 1)
  });

  new Chart(document.getElementById('precisionChart'), {
    type:'line',
    data:{ labels: thresholds, datasets: makeLineDatasets('precision') },
    options: baseOptions('precision (of hits, % correct)', 1)
  });

  new Chart(document.getElementById('costChart'), {
    type:'line',
    data:{ labels: thresholds, datasets: makeLineDatasets('cost_saved_pct') },
    options: baseOptions('% cost saved vs. no cache', 100)
  });
}).catch((err) => {
  console.error('Chart rendering failed:', err);
  document.querySelectorAll('.chart-box').forEach(el => {
    el.innerHTML = '<p style="color:var(--noise);font-size:13px;padding:20px;">Chart failed to load (all CDN sources blocked on this network) &mdash; see console.</p>';
  });
});

const legendEl = document.getElementById('sharedLegend');
BACKEND_ORDER.forEach(key=>{
  const span = document.createElement('span');
  span.innerHTML = `<span class="dot" style="background:${COLORS[key]}"></span>${RESULTS.backends[key].name}`;
  legendEl.appendChild(span);
});

const table = document.getElementById('categoryTable');
const targetThreshold = 0.25;
const categories = ["exact","paraphrase","trap","unrelated"];
let rows = `<tr><th>Backend</th>${categories.map(c=>`<th class="num">${c} (fp / total)</th>`).join('')}</tr>`;
BACKEND_ORDER.forEach(key=>{
  const row = RESULTS.backends[key].by_threshold.find(r => Math.abs(r.threshold - targetThreshold) < 1e-9);
  rows += `<tr><td>${RESULTS.backends[key].name}</td>`;
  categories.forEach(cat=>{
    const s = row.category_stats[cat] || {tp:0,fp:0,tn:0,fn:0};
    const total = s.tp + s.fp + s.tn + s.fn;
    const cellClass = cat === 'trap' ? (s.fp > total/2 ? 'trap-cell' : 'ok-cell') : '';
    rows += `<td class="num ${cellClass}">${s.fp} / ${total}</td>`;
  });
  rows += `</tr>`;
});
table.innerHTML = rows;

const thresholdSelect = document.getElementById('calcThreshold');
thresholds.forEach(t => {
  const opt = document.createElement('option');
  opt.value = t;
  opt.textContent = t.toFixed(2);
  if (Math.abs(t - 0.25) < 1e-9) opt.selected = true;
  thresholdSelect.appendChild(opt);
});

function renderCalculator() {
  const volume = parseFloat(document.getElementById('calcVolume').value) || 0;
  const costPerCall = parseFloat(document.getElementById('calcCost').value) || 0;
  const threshold = parseFloat(thresholdSelect.value);
  const annualNoCache = volume * 365 * costPerCall;

  let rows = '<tr><th>Backend</th><th class="num">% cost saved (measured)</th><th class="num">Est. $ saved/year</th></tr>';
  BACKEND_ORDER.forEach(key => {
    const row = RESULTS.backends[key].by_threshold.find(r => Math.abs(r.threshold - threshold) < 1e-9);
    const pct = row.cost_saved_pct;
    const dollarsSaved = annualNoCache * (pct / 100);
    rows += `<tr><td>${RESULTS.backends[key].name}</td><td class="num">${pct.toFixed(1)}%</td>` +
            `<td class="num ok-cell">$${Math.round(dollarsSaved).toLocaleString()}</td></tr>`;
  });
  document.getElementById('calcTable').innerHTML = rows;
}

['calcVolume', 'calcCost'].forEach(id => document.getElementById(id).addEventListener('input', renderCalculator));
thresholdSelect.addEventListener('change', renderCalculator);
renderCalculator();

fetch('/api/canonical').then(r => r.json()).then(topics => {
  document.getElementById('liveTopics').innerHTML =
    '<b style="color:var(--text)">Already cached:</b> ' +
    topics.map(t => t.text).join(' &nbsp;&middot;&nbsp; ');
}).catch(() => {
  document.getElementById('liveTopics').textContent = '(canonical topic list unavailable -- static file mode)';
});

document.getElementById('liveRunBtn').addEventListener('click', async () => {
  const text = document.getElementById('liveQuery').value.trim();
  const threshold = parseFloat(document.getElementById('liveThreshold').value) || 0.25;
  const resultsEl = document.getElementById('liveResults');
  const noteEl = document.getElementById('liveNote');
  if (!text) return;
  resultsEl.innerHTML = '<p class="section-desc">Running...</p>';
  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, threshold})
    });
    if (!resp.ok) throw new Error('request failed');
    const data = await resp.json();
    let html = '<table><tr><th>Backend</th><th>Result</th><th class="num">Similarity</th><th class="num">Latency</th><th>Matched topic</th></tr>';
    BACKEND_ORDER.forEach(key => {
      const r = data.results[key];
      const tierNote = r.tier === 'candidate' ? ' (Echo Check)' : '';
      html += `<tr><td>${r.name}</td><td class="${r.hit ? 'ok-cell' : ''}">${r.hit ? 'HIT' : 'miss'}${tierNote}</td>` +
              `<td class="num">${r.similarity.toFixed(3)}</td><td class="num">${r.latency_ms.toFixed(1)}ms</td>` +
              `<td>${r.matched_topic || '—'}</td></tr>`;
    });
    html += '</table>';
    resultsEl.innerHTML = html;
    noteEl.style.display = 'none';
  } catch (e) {
    resultsEl.innerHTML = '';
    noteEl.style.display = 'block';
    noteEl.textContent = 'Could not reach the live API — start it with: python3 app.py';
  }
});
</script>
</body>
</html>
"""

HTML = HTML.replace("__DATA_JSON__", DATA_JSON)

with open("dashboard.html", "w") as f:
    f.write(HTML)

print("Wrote dashboard.html")
