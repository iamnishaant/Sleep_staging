/* Somnia · Sleep-Staging Console
   Vanilla JS, no build step, no external dependencies. */

'use strict';

/* ── Constants ──────────────────────────────────────────────────────────── */

const STAGES = ['W', 'N1', 'N2', 'N3', 'REM'];
const STAGE_HEX = {
  W: '#c99024', N1: '#189d92', N2: '#3f76d6', N3: '#7a63e8', REM: '#d6539c',
};
const STAGE_NAME = { W: 'Wake', N1: 'N1', N2: 'N2', N3: 'N3', REM: 'REM' };
const MONO = '"Cascadia Code", Consolas, monospace';

const view = document.getElementById('view');

/* ── Small helpers ──────────────────────────────────────────────────────── */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

const num = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d));
const int = (v) => (v == null ? '—' : Math.round(v).toLocaleString());
const pct = (v, d = 1) => (v == null ? '—' : (Number(v) * 100).toFixed(d) + '%');
const big = (v) => (v == null ? '—' : Math.round(v).toLocaleString());

function duration(minutes) {
  if (minutes == null) return '—';
  const total = Math.round(minutes);
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m`;
}

function bytes(size) {
  if (size == null) return '—';
  if (size > 1e9) return (size / 1e9).toFixed(1) + ' GB';
  if (size > 1e6) return (size / 1e6).toFixed(1) + ' MB';
  if (size > 1e3) return (size / 1e3).toFixed(1) + ' kB';
  return size + ' B';
}

function ago(mtime) {
  if (!mtime) return '';
  const diff = (Date.now() / 1000) - mtime;
  if (diff < 90) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return `${Math.round(diff / 86400)} d ago`;
}

function debounce(fn, wait = 150) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

function badge(text, kind = '') {
  return `<span class="badge ${esc(kind)}">${esc(text)}</span>`;
}

function dot(stateName) {
  return `<span class="dot ${stateName === 'ok' ? 'ok' : stateName === 'warn' ? 'warn' : 'bad'}"></span>`;
}

function stageMix(fractions) {
  const parts = STAGES.map((s) => {
    const w = Math.max(0, (fractions?.[s] || 0) * 100);
    return `<div class="dist-seg" style="width:${w.toFixed(2)}%;background:${STAGE_HEX[s]}" title="${s} ${pct(fractions?.[s])}"></div>`;
  }).join('');
  return `<div class="dist-bar">${parts}</div>`;
}

function loadingCard(label = 'Loading…') {
  return `<div class="splash"><div class="spinner"></div><p>${esc(label)}</p></div>`;
}

function errorCard(error) {
  return `<div class="card"><div class="note bad"><strong>Could not load this view.</strong><br>${esc(error?.message || error)}</div></div>`;
}

function emptyCard(title, body) {
  return `<div class="empty-state"><h3>${esc(title)}</h3><p>${esc(body || '')}</p></div>`;
}

function toast(title, body = '', kind = 'ok') {
  const host = document.getElementById('toasts');
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.innerHTML = `<div class="t-title">${esc(title)}</div>${body ? `<div class="t-body">${esc(body)}</div>` : ''}`;
  host.appendChild(node);
  setTimeout(() => { node.style.opacity = '0'; node.style.transition = 'opacity .25s'; }, 4600);
  setTimeout(() => node.remove(), 5000);
}

/* ── API ────────────────────────────────────────────────────────────────── */

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) { /* ignore */ }
    throw new Error(`${response.status}: ${message}`);
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response.text();
}

function postJSON(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/* ── State ──────────────────────────────────────────────────────────────── */

const state = {
  status: null,
  recs: {},
  split: 'dev',
  search: '',
  cohort: 'all',
  tier: 'all',
  packet: null,
  packetMeta: null,
  poll: null,
  reports: null,
};

/* ── Status chips / header ──────────────────────────────────────────────── */

function setHead(title, sub) {
  document.getElementById('page-title').textContent = title;
  document.getElementById('page-sub').textContent = sub || '';
  document.title = `${title} · Somnia`;
}

function setActiveNav(page) {
  $$('.nav-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.page === page);
  });
}

async function loadStatus(force = false) {
  if (!state.status || force) state.status = await api('/api/status');
  renderChips();
  return state.status;
}

function renderChips() {
  const host = document.getElementById('status-chips');
  const s = state.status;
  if (!s) { host.innerHTML = ''; return; }
  const integrity = s.checks.find((c) => c.id === 'verifier');
  const llama = s.llama_alive;
  host.innerHTML = `
    <span class="chip"><span class="dot accent"></span><b>student_N4kd</b></span>
    <span class="chip"><span class="dot ${llama ? 'ok' : 'warn'}"></span>${llama ? 'Llama online' : 'Llama offline'}</span>
    <span class="chip"><span class="dot ${integrity?.state === 'ok' ? 'ok' : 'warn'}"></span>${integrity?.state === 'ok' ? 'verifier frozen' : 'verify'}</span>
  `;
}

async function getRecs(split) {
  if (!state.recs[split]) {
    const payload = await api(`/api/recordings?split=${encodeURIComponent(split)}`);
    state.recs[split] = payload.recordings;
  }
  return state.recs[split];
}

/* ── Router ─────────────────────────────────────────────────────────────── */

const ROUTES = ['overview', 'recordings', 'recording', 'new', 'reports', 'system'];

async function route() {
  stopPolling();
  const hash = location.hash.replace(/^#\/?/, '');
  const parts = hash.split('/').filter(Boolean);
  const page = parts[0] || 'overview';
  setActiveNav(page === 'recording' ? 'recordings' : (ROUTES.includes(page) ? page : 'overview'));

  try {
    if (page === 'recordings') await pageRecordings(parts[1] || state.split);
    else if (page === 'recording') await pageRecording(parts[1] || 'dev', decodeURIComponent(parts[2] || ''));
    else if (page === 'new') await pageNew();
    else if (page === 'reports') await pageReports(parts[1] ? decodeURIComponent(parts[1]) : null, parts[2] || null);
    else if (page === 'system') await pageSystem();
    else await pageOverview();
  } catch (error) {
    view.innerHTML = errorCard(error);
  }
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

/* ── Overview ───────────────────────────────────────────────────────────── */

async function pageOverview() {
  const s = await loadStatus();
  const m = s.metrics;
  const counts = s.counts;
  const evidence = m.evidence || { items: 19, safe: 5 };
  setHead('Overview', 'Staging → evidence packet → verification → report');

  const kpi = (label, value, unit, foot, accent = false) => `
    <div class="kpi ${accent ? 'accent' : ''}">
      <div class="k-label">${esc(label)}</div>
      <div class="k-value">${value}${unit ? `<small>${esc(unit)}</small>` : ''}</div>
      <div class="k-foot">${foot}</div>
    </div>`;

  const pipeline = [
    ['01', 'Preprocess', 'EDF → 30-second EEG epochs + 34 spectral features'],
    ['02', 'Stage', 'student_N4kd · 139K params · calibrated 5-class probabilities'],
    ['03', 'Packet', 'Clinic metrics, reliability tiers and 19 evidence items'],
    ['04', 'Report', 'Llama-Student constrained by the frozen grammar'],
    ['05', 'Verify', '30 deterministic rules — a passing report or human review'],
  ].map(([n, name, desc]) => `
    <div class="pipe-step">
      <div class="pipe-num">${n}</div>
      <div class="pipe-name">${esc(name)}</div>
      <div class="pipe-desc">${esc(desc)}</div>
    </div>`).join('');

  const f1Rows = (m.per_class || []).map((row) => {
    const width = Math.max(2, (row.f1 || 0) * 100);
    return `
      <div class="bar-row">
        <div class="bar-label"><span class="stage-dot" style="background:${STAGE_HEX[row.stage]}"></span>${esc(row.stage)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${STAGE_HEX[row.stage]}"></div></div>
        <div class="bar-value">${num(row.f1, 3)}</div>
      </div>`;
  }).join('');

  const modelRows = (m.students || []).map((row) => `
    <tr>
      <td><div class="cell-title mono">${esc(row.name)}</div>${row.shipped ? '<div class="cell-sub">shipped model</div>' : ''}</td>
      <td class="num">${int(row.parameters)}</td>
      <td class="num">${num(row.kappa, 4)}</td>
      <td class="num">${num(row.macro_f1, 4)}</td>
      <td class="num">${row.compression ? row.compression.toFixed(2) + '×' : '—'}</td>
    </tr>`).join('');

  const checkRows = s.checks.map((c) => `
    <div class="check-row">
      <span class="dot ${c.state === 'ok' ? 'ok' : c.state === 'warn' ? 'warn' : 'bad'}"></span>
      <div class="c-main">
        <div class="c-label">${esc(c.label)}</div>
        <div class="c-detail">${esc(c.detail)}</div>
      </div>
      <span class="c-state ${c.state}">${c.state === 'ok' ? 'ready' : c.state}</span>
    </div>`).join('');

  view.innerHTML = `
    <div class="stack">
      <section class="card hero">
        <div>
          <div class="eyebrow">Explainable deep learning · Sleep-EDFx</div>
          <h2>One night of EEG becomes a <em>verified</em> sleep report.</h2>
          <p class="hero-lede">
            A compact student model stages every 30 seconds, a calibrated evidence packet records
            how much each number can be trusted, and a deterministic verifier lets the language
            model's report through only when every claim checks out.
          </p>
          <div class="hero-actions">
            <a class="btn btn-primary" href="#/recordings">Explore recordings</a>
            <a class="btn" href="#/new">Analyse a new night</a>
            <a class="btn btn-ghost" href="#/system">System status</a>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-stat"><span class="k">Held-out test κ</span><span class="v">${num(m.kappa, 4)}</span></div>
          <div class="hero-stat"><span class="k">Macro F1</span><span class="v">${num(m.macro_f1, 4)}</span></div>
          <div class="hero-stat"><span class="k">Report model</span><span class="v">${esc(m.report_model || 'Llama-Student')}</span></div>
          <div class="hero-stat"><span class="k">Test reports</span><span class="v">${esc(m.frozen_test?.test_5_of_5 || '29/29')}</span></div>
        </div>
      </section>

      <div class="kpi-grid">
        ${kpi('Parameters', int(m.parameters), '', `${num(m.compression, 2)}× smaller than the teacher`)}
        ${kpi('Test κ', num(m.kappa, 4), '', `accuracy ${pct(m.accuracy, 2)}`, true)}
        ${kpi('Macro F1', num(m.macro_f1, 4), '', `weighted ${num(m.weighted_f1, 4)}`)}
        ${kpi('Evidence items', evidence.items, 'per night', `${evidence.safe} safe to assert`)}
        ${kpi('Recordings', int(counts.dev + counts.test + counts.train), '', `${counts.dev} dev · ${counts.test} test · ${counts.train} train`)}
        ${kpi('Verified reports', int(counts.reports), '', `${counts.review} in human review`)}
      </div>

      <div class="grid grid-2">
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">The pipeline</div><div class="card-sub">Every stage has one job and hands a typed artifact to the next.</div></div>
          </div>
          <div class="pipeline">${pipeline}</div>
          <div class="note mt-16">
            The verifier is frozen and deterministic. The language model decides how to say things,
            never what is true: values come from the packet and each claim is checked against it.
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Per-stage reliability</div><div class="card-sub">Held-out test F1 · ${int(m.n_test_epochs)} epochs</div></div>
            ${badge('test split', 'outline')}
          </div>
          ${f1Rows}
          <div class="note mt-16">N1 is representation-bound and remains the weakest class. The packet flags low-confidence N1 calls rather than promising better accuracy.</div>
        </div>
      </div>

      <div class="grid grid-2">
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Model comparison</div><div class="card-sub">Raw outputs on the locked test subjects</div></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Model</th><th class="num">Params</th><th class="num">κ</th><th class="num">Macro F1</th><th class="num">Compression</th></tr></thead>
              <tbody>${modelRows}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Runtime checks</div><div class="card-sub">Frozen components, staging checkpoint and local report model</div></div>
          </div>
          <div class="check-list">${checkRows}</div>
        </div>
      </div>
    </div>`;
}

/* ── Recordings list ────────────────────────────────────────────────────── */

async function pageRecordings(split) {
  if (!['dev', 'test', 'train', 'new'].includes(split)) split = 'dev';
  state.split = split;
  setHead('Recordings', 'Evidence packets available for inspection and reporting');
  view.innerHTML = loadingCard('Loading recordings…');
  await getRecs(split);

  const s = state.status || await loadStatus();
  const counts = s.counts;

  const tab = (key, label) => {
    const count = state.recs[key] ? state.recs[key].length : (counts[key] ?? 0);
    return `
    <button class="seg ${split === key ? 'active' : ''}" data-split="${key}">
      ${label}<span class="count">${count}</span>
    </button>`;
  };

  view.innerHTML = `
    <div class="stack">
      <div class="row-between wrap">
        <div class="segmented">${tab('dev', 'Dev')}${tab('test', 'Test')}${tab('train', 'Train')}${tab('new', 'UI nights')}</div>
        <div class="toolbar">
          <label class="search">
            <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>
            <input id="rec-search" type="search" placeholder="Search recording or subject…" value="${esc(state.search)}" />
          </label>
          <select id="rec-cohort" class="filter">
            <option value="all">All cohorts</option>
            <option value="SC">SC · healthy ageing</option>
            <option value="ST">ST · temazepam study</option>
          </select>
          <select id="rec-tier" class="filter">
            <option value="all">All nights</option>
            <option value="high">High confidence</option>
            <option value="medium">Medium confidence</option>
            <option value="low">Low confidence</option>
          </select>
        </div>
      </div>

      <div class="card" style="padding:0;overflow:hidden">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Recording</th><th>Subject</th><th>Cohort</th><th>Duration</th>
                <th>Night</th><th class="num">N1 flagged</th><th>Stage mix</th><th>Report</th>
              </tr>
            </thead>
            <tbody id="rec-body"></tbody>
          </table>
        </div>
        <div id="rec-empty"></div>
      </div>
      <div class="row-between">
        <span class="faint mono" id="rec-count"></span>
        <span class="faint">${split === 'test' ? 'Locked split — read once at the end. It is never used to choose anything.' : split === 'dev' ? 'Validation split — every reliability table is fitted here.' : split === 'new' ? 'Nights built from the console. They live in ui/runtime/packets.' : 'Training split — used to train the staging and report models.'}</span>
      </div>
    </div>`;

  $('#rec-cohort').value = state.cohort;
  $('#rec-tier').value = state.tier;

  const render = () => {
    const query = state.search.trim().toLowerCase();
    const rows = state.recs[split].filter((r) => {
      if (state.cohort !== 'all' && r.cohort !== state.cohort) return false;
      if (state.tier !== 'all' && r.tier !== state.tier) return false;
      if (!query) return true;
      return `${r.recording_id} ${r.subject_id}`.toLowerCase().includes(query);
    });
    const body = rows.map((r) => `
      <tr class="clickable" data-split="${split}" data-rec="${esc(r.recording_id)}">
        <td>
          <div class="cell-title mono">${esc(r.recording_id)}</div>
          <div class="cell-sub">${int(r.n_epochs)} epochs</div>
        </td>
        <td class="mono">${esc(r.subject_id)}</td>
        <td>${badge(r.cohort || '—', 'outline')}</td>
        <td>${duration(r.duration_min)}</td>
        <td>${r.tier ? badge(`${r.tier}`, r.tier) : '—'}</td>
        <td class="num">${r.n1_flagged ?? '—'}<span class="faint">/${r.n1_total ?? '—'}</span></td>
        <td style="min-width:130px">${stageMix(r.stage_fractions)}</td>
        <td>${r.has_report ? badge('verified', 'ok') : '<span class="faint">—</span>'}</td>
      </tr>`).join('');
    const bodyEl = $('#rec-body');
    if (bodyEl) bodyEl.innerHTML = body;
    const empty = $('#rec-empty');
    if (empty) empty.innerHTML = rows.length ? '' : emptyCard('No recordings match', 'Adjust the search or filters.');
    const count = $('#rec-count');
    if (count) count.textContent = `${rows.length} of ${state.recs[split].length} recordings`;
  };

  $$('.seg[data-split]').forEach((button) => {
    button.addEventListener('click', () => { location.hash = `#/recordings/${button.dataset.split}`; });
  });
  $('#rec-search').addEventListener('input', debounce((event) => {
    state.search = event.target.value;
    render();
  }, 120));
  $('#rec-cohort').addEventListener('change', (event) => { state.cohort = event.target.value; render(); });
  $('#rec-tier').addEventListener('change', (event) => { state.tier = event.target.value; render(); });
  $('#rec-body').addEventListener('click', (event) => {
    const row = event.target.closest('tr[data-rec]');
    if (row) location.hash = `#/recording/${row.dataset.split}/${encodeURIComponent(row.dataset.rec)}`;
  });

  render();
}

/* ── Recording detail ───────────────────────────────────────────────────── */

async function pageRecording(split, recId) {
  setHead('Recording', `${split} · ${recId}`);
  view.innerHTML = loadingCard(`Loading ${recId}…`);

  const data = await api(`/api/packets/${encodeURIComponent(split)}/${encodeURIComponent(recId)}`);
  state.packet = data.packet;
  state.packetMeta = data;
  const packet = data.packet;
  const tier = packet.night_confidence?.tier || '—';
  const report = data.report;

  view.innerHTML = `
    <a class="back-link" href="#/recordings/${split}">
      <svg viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6"/></svg> All recordings
    </a>
    <div class="stack">
      <div class="card detail-head">
        <div class="detail-title">
          <div class="row wrap">
            ${badge(`${split} split`, 'accent')}
            ${badge(`${tier} night`, tier)}
            ${badge(`schema ${packet.schema_version}`, 'outline')}
            ${packet._ground_truth_withheld ? badge('ground truth withheld', 'outline') : ''}
          </div>
          <h2>${esc(packet.recording_id)}</h2>
          <div class="detail-meta">
            <span class="mono">${esc(packet.subject_id || '—')}</span>
            <span>·</span><span>${esc(packet.cohort || '—')} cohort</span>
            <span>·</span><span>${int(packet.n_epochs)} epochs</span>
            <span>·</span><span>${duration(packet.recording_duration_min)}</span>
            <span>·</span><span>N1 flagged ${packet.n1_confidence_flag?.n_flagged ?? '—'}/${packet.n1_confidence_flag?.n_n1_epochs ?? '—'}</span>
          </div>
        </div>
        <div class="detail-actions">
          ${report ? `<a class="btn" href="#/reports/${encodeURIComponent(report.name)}/${report.source}">View report</a>` : ''}
          <a class="btn" href="${data.download}">Download packet</a>
          <button class="btn btn-primary" id="gen-report">
            <svg viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>
            Generate verified report
          </button>
        </div>
      </div>

      <div id="report-panel"></div>
      ${packetSectionsHTML(packet)}
    </div>`;

  mountPacket(packet, view);

  const button = $('#gen-report');
  button.addEventListener('click', async () => {
    button.disabled = true;
    const panel = $('#report-panel');
    panel.innerHTML = `<div class="card"><div class="row"><div class="spinner"></div><span class="muted">Starting the frozen report pipeline…</span></div></div>`;
    try {
      const { job_id } = await postJSON('/api/jobs/report', { split, recording_id: recId });
      panel.innerHTML = jobCardHTML({ id: job_id, title: `Verified report · ${recId}`, status: 'queued', steps: [], logs: [] });
      pollJob(job_id, {
        update: (job, logs) => { panel.innerHTML = jobCardHTML(job, logs, { report: true }); },
        done: (job) => {
          button.disabled = false;
          if (job.status === 'done' && job.result?.status === 'PASS') {
            delete state.recs[split];
            toast('Report verified', 'Every claim passed the frozen verifier.');
            loadStatus(true);
          } else {
            toast('Verifier blocked the report', 'It was routed to human review.', 'bad');
          }
        },
        error: (err) => {
          button.disabled = false;
          panel.innerHTML = `<div class="card"><div class="note bad">${esc(err.message)}</div></div>`;
        },
      });
    } catch (error) {
      button.disabled = false;
      panel.innerHTML = `<div class="card"><div class="note bad">${esc(error.message)}</div></div>`;
    }
  });
}

/* ── Packet sections (shared by detail and new-night result) ────────────── */

function formatMetric(key, value, pm) {
  if (value == null) return { v: '—', u: '' };
  const isProp = /_Proportion$/.test(key) || key === 'Sleep_Efficiency';
  if (isProp) return { v: (Number(value) * 100).toFixed(1), u: '%' };
  if (pm?.unit === 'minutes') return { v: (Number(value) / 60).toFixed(1), u: 'min' };
  if (key === 'Transition_Rate' || key === 'Wake_Interruptions_per_Hour') return { v: Number(value).toFixed(2), u: '/h' };
  if (key === 'Stage_Transitions' || key === 'REM_Periods') return { v: String(Math.round(value)), u: '' };
  if (key === 'Segment_Duration_Variance') return { v: big(value), u: '' };
  if (key === 'Stage_Transition_Entropy') return { v: Number(value).toFixed(2), u: 'nats' };
  if (key === 'Light_Deep_Ratio') return { v: Number(value).toFixed(2), u: '' };
  return { v: Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }), u: '' };
}

function formatMetricError(key, pm) {
  const mae = pm?.mean_abs_error;
  if (mae == null) return '';
  const isProp = /_Proportion$/.test(key) || key === 'Sleep_Efficiency';
  if (isProp) return `± ${(mae * 100).toFixed(1)} pp MAE`;
  if (pm?.unit === 'minutes') return `± ${Number(mae).toFixed(1)} min MAE`;
  return `MAE ${Number(mae).toFixed(2)}`;
}

function metricName(key) {
  return key.replace(/_/g, ' ');
}

function packetSectionsHTML(packet) {
  const dm = packet.derived_metrics || {};
  const reliability = packet.derived_metric_reliability || {};
  const perMetric = reliability.per_metric || {};
  const stage = packet.per_stage || {};
  const evidence = packet.evidence_items || [];
  const quality = packet.attribution_quality || {};
  const attribution = packet.attribution || {};
  const tier = packet.night_confidence?.tier || '—';
  const flag = packet.n1_confidence_flag || {};
  const decoding = packet.decoding || {};

  const metricCard = (key) => {
    const pm = perMetric[key] || {};
    const formatted = formatMetric(key, dm[key], pm);
    const tierClass = pm.tier || 'fragile';
    return `
      <div class="metric tier-${tierClass}">
        <div class="m-head">
          <span class="m-name">${esc(metricName(key))}</span>
          ${badge(tierClass, tierClass)}
        </div>
        <div class="m-val">${esc(formatted.v)}<small>${esc(formatted.u)}</small></div>
        <div class="m-err">${esc(formatMetricError(key, pm))}</div>
      </div>`;
  };

  const metricGroup = (title, keys, tierName, note) => `
    <div class="mb-20">
      <div class="row-between mb-12">
        <div class="row">
          <span class="stage-dot" style="background:${tierName === 'robust' ? 'var(--ok)' : tierName === 'fragile' ? 'var(--warn)' : 'var(--bad)'}"></span>
          <span class="card-title">${esc(title)}</span>
          ${note ? `<span class="faint" style="font-size:11px">${esc(note)}</span>` : ''}
        </div>
        ${badge(String(keys.length), tierName)}
      </div>
      <div class="metric-grid">${keys.map(metricCard).join('') || '<div class="faint">none</div>'}</div>
    </div>`;

  const stageCards = STAGES.map((s) => {
    const row = stage[s] || {};
    return `
      <div class="stage-card" style="--stage-color:${STAGE_HEX[s]}">
        <div class="s-head">
          <span class="s-name"><span class="stage-dot" style="background:${STAGE_HEX[s]}"></span> ${s}</span>
          ${row.model_reliability_tier ? badge(row.model_reliability_tier, row.model_reliability_tier) : ''}
        </div>
        <div class="s-frac">${pct(row.fraction_of_night)}</div>
        <div class="s-meta">${int(row.n_epochs_predicted)} epochs<br>confidence ${num(row.mean_calibrated_confidence, 2)}<br>test F1 ${num(row.model_f1_on_unseen_subjects, 3)}</div>
      </div>`;
  }).join('');

  const safeItems = evidence.filter((e) => e.safe_to_assert);
  const hedgedItems = evidence.filter((e) => !e.safe_to_assert);
  const evidenceRow = (item) => `
    <tr>
      <td>
        <div class="ev-label">${esc(item.label)}</div>
        <div class="ev-id">${esc(item.id)}</div>
        ${item.caveat ? `<details><summary>caveat</summary><div class="ev-caveat">${esc(item.caveat)}</div></details>` : ''}
      </td>
      <td class="ev-value">${item.unit === 'tier'
        ? badge(item.value, item.value)
        : `${esc(item.value)} <span class="faint">${esc(item.unit || '')}</span>`}</td>
      <td>${item.metric_reliability ? badge(item.metric_reliability, item.metric_reliability) : '<span class="faint">—</span>'}</td>
      <td>${item.safe_to_assert ? badge('safe', 'ok') : badge('hedged', 'warn')}</td>
      <td class="num">${item.mean_abs_error != null ? `${num(item.mean_abs_error, 2)}${item.error_unit ? ` <span class="faint">${esc(item.error_unit)}</span>` : ''}` : '<span class="faint">—</span>'}</td>
    </tr>`;

  const attributionStages = STAGES.map((s) => {
    const row = attribution.per_stage?.[s];
    if (!row) return '';
    return `
      <div class="stage-card" style="--stage-color:${STAGE_HEX[s]}">
        <div class="s-head"><span class="s-name"><span class="stage-dot" style="background:${STAGE_HEX[s]}"></span> ${s}</span><span class="faint mono" style="font-size:10.5px">top1 ${pct(row.top1_share)}</span></div>
        <div class="s-meta" style="margin-top:9px">
          ${(row.top3 || []).map((f) => `<div class="row-between" style="margin:3px 0"><span class="mono">${esc(f.feature)}</span><span class="mono faint">${pct(f.share)}</span></div>`).join('')}
        </div>
      </div>`;
  }).join('');

  const tstKey = 'Total_Sleep_Time';
  const seKey = 'Sleep_Efficiency';
  const wasoKey = 'Wake_After_Sleep_Onset';
  const remKey = 'REM_Latency';

  const miniStat = (key, label) => {
    const f = formatMetric(key, dm[key], perMetric[key]);
    const tierName = perMetric[key]?.tier || '';
    return `
      <div class="mini-stat">
        <div class="row-between"><span class="k">${esc(label)}</span>${tierName ? badge(tierName, tierName) : ''}</div>
        <div class="v">${esc(f.v)}<small>${esc(f.u)}</small></div>
        <div class="m">${esc(formatMetricError(key, perMetric[key]))}</div>
      </div>`;
  };

  return `
    <div class="stat-strip">
      ${miniStat(tstKey, 'Total sleep time')}
      ${miniStat(seKey, 'Sleep efficiency')}
      ${miniStat(wasoKey, 'Wake after sleep onset')}
      ${miniStat(remKey, 'REM latency')}
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Hypnogram</div>
          <div class="card-sub">Decoded stages · ${int(packet.n_epochs)} × 30-second epochs · ${duration(packet.recording_duration_min)}</div>
        </div>
        <div class="row wrap" style="gap:16px">
          <div class="legend">
            ${STAGES.map((s) => `<span class="legend-item"><span class="legend-line" style="background:${STAGE_HEX[s]}"></span>${s}</span>`).join('')}
          </div>
          <label class="check" style="font-size:11.5px">
            <input type="checkbox" id="toggle-flags" checked /> N1 flags
          </label>
        </div>
      </div>
      <div class="hypno-wrap">
        <canvas class="hypno" data-flags="1" role="img" aria-label="Hypnogram of predicted sleep stages"></canvas>
        <canvas class="confidence" role="img" aria-label="Calibrated confidence per epoch"></canvas>
        <div class="hypno-tooltip" id="hypno-tip"></div>
      </div>
      <div class="row-between mt-12 wrap">
        <div class="faint" style="font-size:11.5px">
          Step line: predicted stage. Shaded line below: calibrated confidence (max probability).
          ${flag.n_flagged ? `Amber dots mark ${flag.n_flagged} low-confidence N1 epochs (threshold ${num(flag.threshold, 3)}).` : ''}
        </div>
        <div class="faint mono" style="font-size:10.5px">${esc(decoding.description || '')}</div>
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <div class="card-head">
          <div><div class="card-title">Stage distribution</div><div class="card-sub">Predicted epochs · calibrated confidence · test F1</div></div>
        </div>
        <div class="stage-grid">${stageCards}</div>
      </div>

      <div class="card">
        <div class="card-head">
          <div><div class="card-title">Confidence & evidence quality</div><div class="card-sub">What this night can and cannot support</div></div>
        </div>
        <dl class="kv">
          <dt>Night tier</dt><dd>${badge(tier, tier)}</dd>
          <dt>Mean entropy</dt><dd>${num(packet.night_confidence?.mean_entropy_nats, 3)} nats</dd>
          <dt>Tier boundaries</dt><dd>high ≤ ${num(packet.night_confidence?.tier_boundaries_nats?.high_max, 3)} · medium ≤ ${num(packet.night_confidence?.tier_boundaries_nats?.medium_max, 3)}</dd>
          <dt>N1 flag</dt><dd>${int(flag.n_flagged)} of ${int(flag.n_n1_epochs)} N1 epochs (${pct(flag.fraction_of_n1_flagged)})</dd>
          <dt>Decoder</dt><dd>${esc(decoding.description || 'argmax')} · ${int(decoding.epochs_changed_vs_argmax)} epochs changed</dd>
          <dt>Attribution</dt><dd>${quality.verdict ? badge(quality.verdict, quality.verdict === 'PASS' ? 'ok' : 'warn') : '<span class="faint">not available</span>'} ${quality.n_met != null ? `· ${quality.n_met}/5 predictions met` : ''}</dd>
          <dt>Safe evidence</dt><dd>${safeItems.length} of ${evidence.length} items</dd>
        </dl>
        ${quality.scope ? `<div class="note mt-16">${esc(quality.scope)}</div>` : ''}
        ${flag.how_to_read ? `<div class="note mt-12">${esc(flag.how_to_read)}</div>` : ''}
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div><div class="card-title">Derived clinic metrics</div><div class="card-sub">Fitted reliability on the validation split — not on this night</div></div>
      </div>
      ${metricGroup('Safe to assert', reliability.safe_to_assert || [], 'robust', 'aggregates survive per-epoch error')}
      ${metricGroup('Assert with error bound', reliability.assert_with_error_bound || [], 'fragile', 'state the number with its validation error')}
      ${metricGroup('Do not assert without caveat', reliability.do_not_assert_without_caveat || [], 'unreliable', 'single-event or run-structure metrics')}
      ${reliability.note ? `<div class="note">${esc(reliability.note)}</div>` : ''}
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">Evidence packet</div>
          <div class="card-sub">The only facts a report may cite · ${evidence.length} items · ${safeItems.length} safe, ${hedgedItems.length} hedged</div>
        </div>
        <div class="row" style="gap:6px">${badge('safe', 'ok')}${badge('hedged', 'warn')}</div>
      </div>
      <div class="table-wrap">
        <table class="evidence-table">
          <thead><tr><th>Evidence item</th><th>Value</th><th>Reliability</th><th>Claim type</th><th class="num">Validation MAE</th></tr></thead>
          <tbody>
            ${safeItems.map(evidenceRow).join('')}
            ${hedgedItems.map(evidenceRow).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <div class="card-head">
          <div><div class="card-title">Per-stage attributions</div><div class="card-sub">Integrated Gradients · this recording's own epochs</div></div>
          ${quality.verdict ? badge(`gate 3a ${quality.verdict}`, quality.verdict === 'PASS' ? 'ok' : 'warn') : ''}
        </div>
        ${attributionStages ? `<div class="stage-grid">${attributionStages}</div>` : '<div class="faint">Attribution is null by design for this packet.</div>'}
        ${attribution.how_to_read ? `<div class="note mt-16">${esc(attribution.how_to_read)}</div>` : ''}
      </div>

      <div class="card">
        <div class="card-head">
          <div><div class="card-title">Limitations</div><div class="card-sub">Shipped with the packet, not appended afterwards</div></div>
        </div>
        <ul class="clean">${(packet.limitations || []).map((line) => `<li>${esc(line)}</li>`).join('')}</ul>
        <details class="mt-16">
          <summary class="faint" style="cursor:pointer;font-size:12px">Packet provenance</summary>
          <dl class="kv mt-12">
            <dt>Model</dt><dd>${esc(packet.provenance?.model)}</dd>
            <dt>Checkpoint</dt><dd>${esc(packet.provenance?.model_checkpoint)}</dd>
            <dt>SHA-256</dt><dd>${esc((packet.provenance?.model_sha256 || '').slice(0, 24))}…</dd>
            <dt>Temperature</dt><dd>${num(packet.provenance?.calibration_temperature, 4)}</dd>
            <dt>Decoder</dt><dd>${esc(packet.provenance?.hypnogram_decoder)}</dd>
            <dt>Generated</dt><dd>${esc(packet.provenance?.generated_at)}</dd>
            <dt>Schema</dt><dd>${esc(packet.schema_version)}</dd>
          </dl>
        </details>
      </div>
    </div>`;
}

/* ── Canvas: hypnogram ──────────────────────────────────────────────────── */

function drawHypno(canvas, packet) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = window.devicePixelRatio || 1;
  const W = rect.width;
  const H = rect.height || 290;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const stages = packet.hypnogram?.predicted_stages || [];
  const n = stages.length || packet.n_epochs || 1;
  const pad = { l: 50, r: 20, t: 16, b: 28 };
  const order = ['W', 'REM', 'N1', 'N2', 'N3'];
  const plotW = Math.max(10, W - pad.l - pad.r);
  const plotH = Math.max(10, H - pad.t - pad.b);
  const yFor = (stage) => pad.t + Math.max(0, order.indexOf(stage)) * (plotH / (order.length - 1));
  const xFor = (index) => pad.l + (index / Math.max(1, n - 1)) * plotW;

  /* horizontal guides + labels */
  ctx.font = `600 10px ${MONO}`;
  order.forEach((stage) => {
    const y = yFor(stage);
    ctx.strokeStyle = 'rgba(46, 40, 28, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, y + 0.5);
    ctx.lineTo(W - pad.r, y + 0.5);
    ctx.stroke();
    ctx.fillStyle = '#9b968b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(stage, pad.l - 11, y);
  });

  /* time grid */
  const totalHours = n / 120;
  const stepH = totalHours <= 4 ? 0.5 : totalHours <= 10 ? 1 : 2;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#9b968b';
  ctx.font = `10px ${MONO}`;
  for (let h = 0; h <= totalHours + 0.001; h += stepH) {
    const epoch = h * 120;
    if (epoch > n - 1) break;
    const x = xFor(epoch);
    ctx.strokeStyle = 'rgba(46, 40, 28, 0.05)';
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, H - pad.b);
    ctx.stroke();
    const label = stepH < 1 ? `${Math.floor(h)}:${h % 1 ? '30' : '00'}` : `${h}h`;
    ctx.fillText(label, x, H - pad.b + 8);
  }

  /* stage runs */
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && stages[j + 1] === stages[i]) j += 1;
    const stage = stages[i];
    const colour = STAGE_HEX[stage] || '#8a8375';
    const y = yFor(stage);
    const x0 = xFor(Math.max(0, i - 0.5));
    const x1 = xFor(Math.min(n - 1, j + 0.5));

    ctx.strokeStyle = colour;
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();

    if (j + 1 < n) {
      const y2 = yFor(stages[j + 1]);
      ctx.strokeStyle = colour;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x1, y);
      ctx.lineTo(x1, y2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    i = j + 1;
  }

  /* low-confidence N1 flags */
  if (canvas.dataset.flags !== '0') {
    const flagged = new Set(packet.n1_confidence_flag?.flagged_epochs || []);
    flagged.forEach((index) => {
      if (stages[index] !== 'N1') return;
      ctx.fillStyle = '#b57d16';
      ctx.beginPath();
      ctx.arc(xFor(index), yFor('N1'), 2, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  canvas._packet = packet;
  canvas._geom = { n, pad, plotW, xFor, yFor, stages };
}

function drawConfidence(canvas, packet) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = window.devicePixelRatio || 1;
  const W = rect.width;
  const H = rect.height || 104;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const rows = packet.probabilities?.calibrated || [];
  const n = rows.length || packet.n_epochs || 1;
  if (!rows.length) {
    ctx.fillStyle = '#9b968b';
    ctx.font = `11px ${MONO}`;
    ctx.fillText('Calibrated probabilities are not present in this packet.', 10, H / 2);
    return;
  }
  const pad = { l: 50, r: 20, t: 8, b: 16 };
  const plotW = Math.max(10, W - pad.l - pad.r);
  const plotH = Math.max(10, H - pad.t - pad.b);
  const xFor = (index) => pad.l + (index / Math.max(1, n - 1)) * plotW;
  const yFor = (prob) => pad.t + (1 - Math.max(0, Math.min(1, prob))) * plotH;

  /* threshold */
  const threshold = packet.n1_confidence_flag?.threshold || 0.575;
  ctx.strokeStyle = 'rgba(181, 125, 22, 0.4)';
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, yFor(threshold));
  ctx.lineTo(W - pad.r, yFor(threshold));
  ctx.stroke();
  ctx.setLineDash([]);

  const values = rows.map((row) => Math.max(...row));

  /* fill */
  const gradient = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
  gradient.addColorStop(0, 'rgba(68, 87, 196, 0.18)');
  gradient.addColorStop(1, 'rgba(68, 87, 196, 0.01)');
  ctx.beginPath();
  ctx.moveTo(xFor(0), H - pad.b);
  values.forEach((v, index) => ctx.lineTo(xFor(index), yFor(v)));
  ctx.lineTo(xFor(n - 1), H - pad.b);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  /* line */
  ctx.beginPath();
  values.forEach((v, index) => {
    const x = xFor(index);
    const y = yFor(v);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#4457c4';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  /* labels */
  ctx.fillStyle = '#9b968b';
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('1.0', pad.l - 10, pad.t + 3);
  ctx.fillText('0.5', pad.l - 10, yFor(0.5));
  ctx.fillText('0', pad.l - 10, H - pad.b - 2);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('confidence', pad.l + 4, pad.t - 6);
  ctx.textAlign = 'right';
  ctx.fillText(`N1 threshold ${threshold.toFixed(3)}`, W - pad.r, pad.t - 6);
  canvas._packet = packet;
}

function attachHypnoHover(canvas, tooltip) {
  canvas.addEventListener('mousemove', (event) => {
    const geom = canvas._geom;
    const packet = canvas._packet;
    if (!geom || !packet) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let index = Math.round(((x - geom.pad.l) / geom.plotW) * (geom.n - 1));
    index = Math.max(0, Math.min(geom.n - 1, index));
    const stage = geom.stages[index];
    if (!stage) return;
    const row = packet.probabilities?.calibrated?.[index];
    const confidence = row ? Math.max(...row) : null;
    const minutes = index * 0.5;
    const hh = Math.floor(minutes / 60);
    const mm = Math.floor(minutes % 60);
    const flagged = new Set(packet.n1_confidence_flag?.flagged_epochs || []).has(index);

    tooltip.innerHTML = `
      <div class="tt-stage"><span class="stage-dot" style="background:${STAGE_HEX[stage]}"></span>${esc(STAGE_NAME[stage] || stage)}${flagged ? ' · flagged' : ''}</div>
      <div class="tt-meta">epoch ${index} · ${hh}h ${String(mm).padStart(2, '0')}m</div>
      <div class="tt-meta">confidence ${confidence == null ? '—' : confidence.toFixed(3)}</div>`;
    tooltip.style.left = `${Math.max(60, Math.min(rect.width - 60, geom.xFor(index)))}px`;
    tooltip.style.top = `${y - 6}px`;
    tooltip.style.opacity = '1';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });
}

function mountPacket(packet, root) {
  const hypno = $('canvas.hypno', root);
  const confidence = $('canvas.confidence', root);
  const tooltip = $('.hypno-tooltip', root);
  if (hypno) {
    drawHypno(hypno, packet);
    if (tooltip) attachHypnoHover(hypno, tooltip);
    const toggle = $('#toggle-flags', root);
    if (toggle) {
      toggle.addEventListener('change', () => {
        hypno.dataset.flags = toggle.checked ? '1' : '0';
        drawHypno(hypno, packet);
      });
    }
  }
  if (confidence) drawConfidence(confidence, packet);
}

const redrawCanvases = debounce(() => {
  $$('canvas.hypno').forEach((canvas) => {
    if (canvas._packet) drawHypno(canvas, canvas._packet);
  });
  $$('canvas.confidence').forEach((canvas) => {
    if (canvas._packet) drawConfidence(canvas, canvas._packet);
  });
}, 140);
window.addEventListener('resize', redrawCanvases);

/* ── Jobs ───────────────────────────────────────────────────────────────── */

function jobStepsHTML(job) {
  return `<div class="steps">${(job.steps || []).map((step) => `
    <div class="step ${step.status}">
      <span class="s-ind">${step.status === 'done' ? '✓' : step.status === 'failed' ? '!' : step.status === 'skipped' ? '–' : ''}</span>
      <span>${esc(step.label)}</span>
      <span class="s-state">${esc(step.status)}</span>
    </div>`).join('')}</div>`;
}

function jobLogHTML(logs) {
  const lines = (logs || []).slice(-400);
  if (!lines.length) return '<div class="empty">Waiting for output…</div>';
  return lines.map((line) => `
    <div class="line ${line.level || 'info'}">
      <span class="ts">${esc(line.t)}</span>
      <span class="msg">${esc(line.msg)}</span>
    </div>`).join('');
}

function jobCardHTML(job, logs = [], options = {}) {
  const status = job.status || 'queued';
  const kind = status === 'done' ? (job.result?.status === 'PASS' ? 'ok' : 'warn')
    : status === 'failed' ? 'bad' : 'accent';
  let extra = '';
  if (status === 'done' && options.report) {
    const result = job.result || {};
    if (result.status === 'PASS' && result.report_text) {
      extra = `
        <div class="row-between mt-16 mb-12 wrap">
          <div class="row">${badge('verifier PASS', 'ok')}<span class="faint" style="font-size:11.5px">${esc(result.audit?.recording_id || '')} · ${esc(result.audit?.llm_wall_s || '—')} s generation</span></div>
          <div class="row">
            <a class="btn btn-sm" href="/api/reports/file/${encodeURIComponent(result.ui_report || '')}?source=ui&download=1">Download</a>
            <a class="btn btn-sm" href="#/reports">All reports</a>
          </div>
        </div>
        <div class="paper">${esc(result.report_text)}</div>`;
    } else {
      const violations = result.violations || [];
      extra = `
        <div class="mt-16">
          <div class="row mb-12">${badge(`${violations.length} violation${violations.length === 1 ? '' : 's'}`, 'bad')}<span class="faint" style="font-size:11.5px">The renderer refused; the night is queued for human review.</span></div>
          ${violations.map((v) => `<div class="violation"><span class="v-code">!</span><span>${esc(v)}</span></div>`).join('')}
          ${result.raw_output ? `<details class="mt-12"><summary class="faint" style="cursor:pointer;font-size:12px">Model output</summary><div class="paper paper-dark mt-12">${esc(result.raw_output)}</div></details>` : ''}
        </div>`;
    }
  }
  if (status === 'failed') {
    extra = `<div class="note bad mt-16">${esc(job.error || 'The job failed.')}</div>`;
  }
  return `
    <div class="card ${options.report ? (job.result?.status === 'PASS' ? 'report-card' : job.result?.status ? 'report-card fail' : '') : ''}">
      <div class="card-head">
        <div><div class="card-title">${esc(job.title || 'Job')}</div><div class="card-sub mono">${esc(job.id || '')}</div></div>
        ${badge(status, kind)}
      </div>
      ${job.steps?.length ? jobStepsHTML(job) : ''}
      <div class="console mt-16" id="job-console">${jobLogHTML(logs)}</div>
      ${extra}
    </div>`;
}

function stopPolling() {
  if (state.poll) {
    clearInterval(state.poll.timer);
    state.poll = null;
  }
}

function pollJob(jobId, handlers) {
  stopPolling();
  const ctx = { timer: null, offset: 0, logs: [] };
  state.poll = ctx;

  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}?offset=${ctx.offset}`);
      ctx.offset = job.next_offset;
      ctx.logs.push(...(job.logs || []));
      handlers.update?.(job, ctx.logs);
      const consoleEl = document.getElementById('job-console');
      if (consoleEl) consoleEl.scrollTop = consoleEl.scrollHeight;
      if (job.status === 'done' || job.status === 'failed') {
        clearInterval(ctx.timer);
        state.poll = null;
        handlers.done?.(job);
      }
    } catch (error) {
      clearInterval(ctx.timer);
      state.poll = null;
      handlers.error?.(error);
    }
  };

  ctx.timer = setInterval(tick, 950);
  tick();
}

/* ── New analysis ───────────────────────────────────────────────────────── */

async function pageNew() {
  setHead('New analysis', 'Raw EDF → staging → evidence packet → verified report');
  const s = await loadStatus();

  view.innerHTML = `
    <div class="stack">
      <div class="grid grid-2">
        <div class="card stack">
          <div>
            <div class="eyebrow">Raw PSG</div>
            <h2 style="font-size:22px;letter-spacing:-.03em;margin-top:8px">Analyse a new night</h2>
            <p class="muted mt-8" style="font-size:13px">
              The console calls the project's own pipeline. It reads the EDF the same way the
              command-line deployment does and writes only to <span class="mono">ui/runtime/</span>.
            </p>
          </div>

          <div class="dropzone" id="dropzone">
            <div class="dz-icon">
              <svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>
            </div>
            <h3>Drop a PSG recording</h3>
            <p>.edf or .rec from Sleep-EDFx. A companion hypnogram in the same folder is optional — the packet is built for reporting either way.</p>
            <span class="btn btn-sm">Choose file</span>
          </div>
          <input id="file-input" type="file" accept=".edf,.rec" hidden />
          <div id="file-pill"></div>

          <label class="check">
            <input type="checkbox" id="make-report" checked />
            Generate the verified report (requires llama-server on port 8080)
          </label>

          <div class="row-between">
            <button class="btn btn-primary" id="run-pipeline" disabled>
              <svg viewBox="0 0 24 24"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>
              Run pipeline
            </button>
            <span class="faint mono" style="font-size:11px">~115 s for the report step</span>
          </div>

          <div class="note ${s.llama_alive ? '' : 'warn'}">
            ${s.llama_alive
              ? `llama-server is online${s.checks.find((c) => c.id === 'llama')?.detail ? ` — ${esc(s.checks.find((c) => c.id === 'llama').detail)}` : ''}.`
              : 'llama-server is offline. Preprocessing, staging and packet building still work; the report step will be skipped. Start the server to enable it.'}
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Pipeline activity</div><div class="card-sub">Stages and live output</div></div>
            <span id="job-badge"></span>
          </div>
          <div id="job-panel">
            <div class="steps">
              ${['Preprocess EDF', 'Stage with student_N4kd', 'Build evidence packet', 'Generate verified report']
                .map((label, index) => `<div class="step"><span class="s-ind">${index + 1}</span><span>${esc(label)}</span><span class="s-state">pending</span></div>`).join('')}
            </div>
            <div class="console mt-16"><div class="empty">No run yet. Choose a recording and press Run pipeline.</div></div>
          </div>
        </div>
      </div>

      <div id="edf-result"></div>
    </div>`;

  let file = null;
  const dropzone = $('#dropzone');
  const input = $('#file-input');
  const pill = $('#file-pill');
  const runButton = $('#run-pipeline');

  const selectFile = (chosen) => {
    file = chosen;
    runButton.disabled = !file;
    pill.innerHTML = file ? `
      <div class="file-pill">
        <div><div class="f-name">${esc(file.name)}</div><div class="f-size">${bytes(file.size)}</div></div>
        <button class="btn btn-ghost btn-sm" id="clear-file">Clear</button>
      </div>` : '';
    const clear = $('#clear-file');
    if (clear) clear.addEventListener('click', () => { file = null; input.value = ''; selectFile(null); });
  };

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('drag');
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) selectFile(dropped);
  });
  input.addEventListener('change', () => { if (input.files?.[0]) selectFile(input.files[0]); });

  runButton.addEventListener('click', async () => {
    if (!file) return;
    runButton.disabled = true;
    const makeReport = $('#make-report').checked;
    const panel = $('#job-panel');
    const resultHost = $('#edf-result');
    resultHost.innerHTML = '';
    panel.innerHTML = '<div class="row"><div class="spinner"></div><span class="muted">Uploading recording…</span></div>';

    try {
      const response = await fetch(`/api/uploads?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: file,
      });
      if (!response.ok) throw new Error(`Upload failed (${response.status})`);
      const upload = await response.json();

      const { job_id } = await postJSON('/api/jobs/edf', {
        upload_id: upload.upload_id,
        generate_report: makeReport,
      });

      $('#job-badge').innerHTML = badge('running', 'accent');
      pollJob(job_id, {
        update: (job, logs) => {
          panel.innerHTML = jobCardInner(job, logs);
          const consoleEl = document.getElementById('job-console');
          if (consoleEl) consoleEl.scrollTop = consoleEl.scrollHeight;
          $('#job-badge').innerHTML = badge(job.status, job.status === 'failed' ? 'bad' : job.status === 'done' ? 'ok' : 'accent');
        },
        done: (job) => {
          runButton.disabled = false;
          const result = job.result || {};
          if (result.packet) {
            renderEdfResult(result, resultHost);
            delete state.recs.new;
            getRecs('new').catch(() => {});
            loadStatus(true).catch(() => {});
            toast('Night processed', `Packet built for ${result.recording_id}.`,
              result.status === 'FAIL' || result.status === 'RENDER_FAIL' ? 'bad' : 'ok');
          } else if (job.status === 'failed') {
            toast('Pipeline failed', job.error || '', 'bad');
          }
        },
        error: (error) => {
          runButton.disabled = false;
          panel.innerHTML = `<div class="note bad">${esc(error.message)}</div>`;
        },
      });
    } catch (error) {
      runButton.disabled = false;
      panel.innerHTML = `<div class="note bad">${esc(error.message)}</div>`;
      toast('Could not start the pipeline', error.message, 'bad');
    }
  });
}

function jobCardInner(job, logs) {
  const status = job.status;
  return `
    <div class="steps">${(job.steps || []).map((step) => `
      <div class="step ${step.status}">
        <span class="s-ind">${step.status === 'done' ? '✓' : step.status === 'failed' ? '!' : step.status === 'skipped' ? '–' : ''}</span>
        <span>${esc(step.label)}</span>
        <span class="s-state">${esc(step.status)}</span>
      </div>`).join('')}</div>
    <div class="console mt-16" id="job-console">${jobLogHTML(logs)}</div>
    ${status === 'failed' ? `<div class="note bad mt-16">${esc(job.error || '')}</div>` : ''}`;
}

function renderEdfResult(result, host) {
  const packet = result.packet;
  const report = result.report_text ? `
    <div class="card report-card">
      <div class="card-head">
        <div><div class="card-title">Verified report</div><div class="card-sub">${esc(packet.recording_id)} · verifier ${esc(result.status)}</div></div>
        <div class="row">${badge('PASS', 'ok')}<a class="btn btn-sm" href="/api/reports/file/${encodeURIComponent(result.ui_report || '')}?source=ui&download=1">Download</a></div>
      </div>
      <div class="paper">${esc(result.report_text)}</div>
    </div>` : '';

  const note = result.report_note ? `<div class="note warn">${esc(result.report_note)}</div>` : '';

  const blocked = (result.violations || []).length || result.status === 'RENDER_FAIL' ? `
    <div class="card report-card fail">
      <div class="card-head">
        <div><div class="card-title">Report blocked</div><div class="card-sub">The verifier refused to render this report; the packet is still shown below.</div></div>
        ${badge(result.status || 'FAIL', 'bad')}
      </div>
      ${(result.violations || []).map((v) => `<div class="violation"><span class="v-code">!</span><span>${esc(v)}</span></div>`).join('')}
      ${result.raw_output ? `<details class="mt-12"><summary class="faint" style="cursor:pointer;font-size:12px">Model output</summary><div class="paper paper-dark mt-12">${esc(result.raw_output)}</div></details>` : ''}
    </div>` : '';

  host.innerHTML = `
    <div class="stack">
      <div class="card detail-head">
        <div class="detail-title">
          <div class="row wrap">${badge('new night', 'accent')}${badge(`${packet.night_confidence?.tier || '—'} night`, packet.night_confidence?.tier || '')}</div>
          <h2>${esc(packet.recording_id)}</h2>
          <div class="detail-meta">
            <span class="mono">${esc(packet.subject_id || '—')}</span><span>·</span>
            <span>${int(packet.n_epochs)} epochs</span><span>·</span>
            <span>${esc(result.packet_path || '')}</span>
          </div>
        </div>
        ${result.report_path ? `<div class="detail-actions"><span class="faint" style="font-size:11.5px">deploy report: ${esc(result.report_path)}</span></div>` : ''}
      </div>
      ${note}
      ${blocked}
      ${report}
      ${packetSectionsHTML(packet)}
    </div>`;
  mountPacket(packet, host);
}

/* ── Reports ────────────────────────────────────────────────────────────── */

async function pageReports(selectedName = null, selectedSource = null) {
  setHead('Reports', 'Verified reports and the human-review queue');
  const data = await api('/api/reports');
  state.reports = data;

  const reports = data.reports || [];
  const review = data.review || [];
  const selected = reports.find((r) => r.name === selectedName && (!selectedSource || r.source === selectedSource)) || null;

  const listRows = reports.map((row) => `
    <button class="check-row" data-report="${esc(row.name)}" data-source="${esc(row.source)}" style="width:100%;text-align:left;cursor:pointer;background:${selected?.name === row.name ? 'var(--accent-soft)' : ''}">
      <span class="dot ${row.source === 'deploy' ? 'ok' : 'accent'}"></span>
      <div class="c-main">
        <div class="c-label mono">${esc(row.recording_id)}</div>
        <div class="c-detail">${esc(row.date || '')} · ${bytes(row.size)} · ${ago(row.mtime)} · ${esc(row.source)}</div>
      </div>
      <span class="badge ${row.source === 'deploy' ? 'ok' : 'accent'}">${row.source === 'deploy' ? 'frozen' : 'this session'}</span>
    </button>`).join('');

  const reviewRows = review.map((row) => `
    <div class="check-row">
      <span class="dot bad"></span>
      <div class="c-main">
        <div class="c-label mono">${esc(row.recording_id || row.name)}</div>
        <div class="c-detail">${esc(row.timestamp || '')} · ${(row.violations || []).length} violation(s)</div>
      </div>
      <button class="btn btn-ghost btn-sm" data-review="${esc(row.name)}">Inspect</button>
    </div>`).join('');

  view.innerHTML = `
    <div class="grid grid-reports">
      <div class="stack">
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Reports</div><div class="card-sub">${reports.length} file${reports.length === 1 ? '' : 's'} · deploy/reports and ui/runtime/reports</div></div>
            <button class="btn btn-ghost btn-sm" id="refresh-reports">Refresh</button>
          </div>
          <div class="check-list">${listRows || emptyCard('No reports yet', 'Generate one from a recording.')}</div>
        </div>
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Human review</div><div class="card-sub">${review.length} blocked report${review.length === 1 ? '' : 's'}</div></div>
          </div>
          <div class="check-list">${reviewRows || '<div class="faint" style="font-size:12px">Nothing blocked. Every generated report passed the verifier.</div>'}</div>
        </div>
      </div>

      <div class="card" id="report-viewer">
        ${selected ? '<div class="row"><div class="spinner"></div><span class="muted">Loading report…</span></div>' : emptyCard('Select a report', 'Any generated report opens here. The text is the renderer output, not the model output.')}
      </div>
    </div>`;

  $$('[data-report]').forEach((button) => {
    button.addEventListener('click', () => {
      location.hash = `#/reports/${encodeURIComponent(button.dataset.report)}/${button.dataset.source}`;
    });
  });
  $('#refresh-reports').addEventListener('click', () => { state.reports = null; route(); });

  $$('[data-review]').forEach((button) => {
    button.addEventListener('click', async () => {
      const record = await api(`/api/review/${encodeURIComponent(button.dataset.review)}`);
      const viewer = $('#report-viewer');
      viewer.innerHTML = `
        <div class="card-head">
          <div><div class="card-title">Human review · ${esc(record.recording_id || '')}</div><div class="card-sub">${esc(record.timestamp || '')}</div></div>
          ${badge(`${(record.violations || []).length} violations`, 'bad')}
        </div>
        ${(record.violations || []).map((v) => `<div class="violation"><span class="v-code">!</span><span>${esc(v)}</span></div>`).join('')}
        ${record.raw_output ? `<details class="mt-12" open><summary class="faint" style="cursor:pointer;font-size:12px">Raw model output</summary><div class="paper paper-dark mt-12">${esc(record.raw_output)}</div></details>` : ''}`;
    });
  });

  if (selected) {
    try {
      const text = await api(`/api/reports/file/${encodeURIComponent(selected.name)}?source=${encodeURIComponent(selected.source)}`);
      const viewer = $('#report-viewer');
      viewer.innerHTML = `
        <div class="card-head">
          <div>
            <div class="card-title mono">${esc(selected.recording_id)}</div>
            <div class="card-sub">${esc(selected.date || '')} · ${bytes(selected.size)} · ${esc(selected.source)}</div>
          </div>
          <div class="row">
            <a class="btn btn-sm" href="/api/reports/file/${encodeURIComponent(selected.name)}?source=${selected.source}&download=1">Download</a>
            <a class="btn btn-sm btn-ghost" href="#/recordings">Recordings</a>
          </div>
        </div>
        <div class="paper">${esc(text)}</div>`;
    } catch (error) {
      $('#report-viewer').innerHTML = `<div class="note bad">${esc(error.message)}</div>`;
    }
  }
}

/* ── System ─────────────────────────────────────────────────────────────── */

async function pageSystem() {
  setHead('System', 'Frozen release, runtime checks and artifact paths');
  const s = await loadStatus(true);
  const m = s.metrics;
  const gen = m.generation || {};
  const cfg = m.report_model;

  const checks = s.checks.map((c) => `
    <div class="check-row">
      <span class="dot ${c.state === 'ok' ? 'ok' : c.state === 'warn' ? 'warn' : 'bad'}"></span>
      <div class="c-main">
        <div class="c-label">${esc(c.label)}</div>
        <div class="c-detail">${esc(c.detail)}</div>
      </div>
      <span class="c-state ${c.state}">${c.state}</span>
    </div>`).join('');

  view.innerHTML = `
    <div class="stack">
      <div class="grid grid-2">
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Runtime checks</div><div class="card-sub">Read-only verification against the frozen manifest</div></div>
            <button class="btn btn-sm" id="recheck">Re-check</button>
          </div>
          <div class="check-list" id="check-list">${checks}</div>
          <div id="integrity-note" class="mt-16"></div>
        </div>

        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Frozen release</div><div class="card-sub">deploy/frozen.json · do not edit</div></div>
          </div>
          <dl class="kv">
            <dt>Report model</dt><dd>${esc(cfg || '—')}</dd>
            <dt>Quantisation</dt><dd>Q4_K_M</dd>
            <dt>Temperature</dt><dd>${esc(gen.temperature ?? '—')}</dd>
            <dt>Seed</dt><dd>${esc(gen.seed ?? '—')}</dd>
            <dt>Context</dt><dd>${esc(gen.context_tokens ?? '—')} tokens</dd>
            <dt>Grammar</dt><dd>${esc(gen.grammar || '—')}</dd>
            <dt>Dev 5/5</dt><dd>${esc(m.frozen_test?.dev_5_of_5 || '—')}</dd>
            <dt>Test 5/5</dt><dd>${esc(m.frozen_test?.test_5_of_5 || '—')}</dd>
            <dt>Test violations</dt><dd>${esc(m.frozen_test?.test_violations ?? '—')}</dd>
            <dt>Numeric fidelity</dt><dd>${num(m.frozen_test?.test_numeric_fidelity, 1)}</dd>
          </dl>
        </div>
      </div>

      <div class="grid grid-2">
        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Staging model</div><div class="card-sub">student_N4kd · the model that ships</div></div>
            ${badge('M0', 'accent')}
          </div>
          <dl class="kv">
            <dt>Parameters</dt><dd>${int(m.parameters)}</dd>
            <dt>Compression</dt><dd>${num(m.compression, 2)}× vs teacher</dd>
            <dt>Temperature</dt><dd>${num(m.temperature, 4)}</dd>
            <dt>Test accuracy</dt><dd>${pct(m.accuracy, 2)}</dd>
            <dt>Test κ</dt><dd>${num(m.kappa, 4)}</dd>
            <dt>Macro F1</dt><dd>${num(m.macro_f1, 4)}</dd>
            <dt>Epochs scored</dt><dd>${int(m.n_test_epochs)}</dd>
          </dl>
          <div class="divider"></div>
          <div class="row wrap">
            ${(m.per_class || []).map((row) => `<span class="badge outline"><span class="stage-dot" style="background:${STAGE_HEX[row.stage]}"></span> ${row.stage} ${num(row.f1, 3)}</span>`).join('')}
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div><div class="card-title">Artifact paths</div><div class="card-sub">Everything the console reads</div></div>
          </div>
          <dl class="kv">
            <dt>Dev packets</dt><dd>${esc(s.packet_dirs.dev)}</dd>
            <dt>Test packets</dt><dd>${esc(s.packet_dirs.test)}</dd>
            <dt>Train packets</dt><dd>${esc(s.packet_dirs.train)}</dd>
            <dt>UI nights</dt><dd>${esc(s.packet_dirs.new)}</dd>
            <dt>Reports</dt><dd>deploy/reports</dd>
            <dt>Review queue</dt><dd>deploy/human_review</dd>
            <dt>Audit log</dt><dd>deploy/audit_log.jsonl</dd>
            <dt>UI runtime</dt><dd>${esc(s.runtime_dir)}</dd>
          </dl>
          <div class="note mt-16">
            The console adds no model, no table and no rule. It imports the frozen report tier and
            the existing pipeline; the only new writes are under <span class="mono">ui/runtime/</span>
            plus the deployment's normal report and audit outputs.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div><div class="card-title">Model comparison</div><div class="card-sub">Held-out test subjects · raw outputs</div></div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Model</th><th class="num">Params</th><th class="num">κ</th><th class="num">Macro F1</th><th class="num">Accuracy</th><th class="num">Compression</th></tr></thead>
            <tbody>${(m.students || []).map((row) => `
              <tr>
                <td><div class="cell-title mono">${esc(row.name)}</div>${row.shipped ? '<div class="cell-sub">shipped</div>' : ''}</td>
                <td class="num">${int(row.parameters)}</td>
                <td class="num">${num(row.kappa, 4)}</td>
                <td class="num">${num(row.macro_f1, 4)}</td>
                <td class="num">${pct(row.accuracy, 2)}</td>
                <td class="num">${row.compression ? row.compression.toFixed(2) + '×' : '—'}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>`;

  $('#recheck').addEventListener('click', async () => {
    const note = $('#integrity-note');
    note.innerHTML = '<div class="row"><div class="spinner"></div><span class="muted">Checking hashes…</span></div>';
    try {
      const result = await api('/api/integrity', { method: 'POST' });
      note.innerHTML = result.ok
        ? '<div class="note">All frozen components match <span class="mono">deploy/frozen.json</span>. The verifier, grammar and serializer are unchanged.</div>'
        : `<div class="note bad">${result.checks.filter((c) => c.state !== 'ok').map((c) => esc(`${c.label}: ${c.detail}`)).join('<br>')}</div>`;
      toast(result.ok ? 'Integrity verified' : 'Integrity mismatch', result.ok ? 'Frozen components match.' : 'Check deploy/frozen.json.', result.ok ? 'ok' : 'bad');
      loadStatus(true);
    } catch (error) {
      note.innerHTML = `<div class="note bad">${esc(error.message)}</div>`;
    }
  });
}

/* ── Boot ───────────────────────────────────────────────────────────────── */

window.addEventListener('hashchange', route);

(async function boot() {
  try {
    state.status = await api('/api/status');
    renderChips();
  } catch (error) {
    view.innerHTML = `<div class="card"><div class="note bad">
      <strong>The console cannot reach its API.</strong><br>
      ${esc(error.message)}<br><br>
      Start the server with <span class="mono">python -m ui</span> and reload.
    </div></div>`;
    return;
  }
  if (!location.hash) location.hash = '#/overview';
  route();
})();
