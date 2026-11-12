// Frontend for the local RAG UI: one file, a small init function per page (body[data-page]).
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const fmt = (n) => Number(n).toLocaleString('ru-RU');

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
  return data;
}

// --- side panel ------------------------------------------------------------------------------

$('#nav-toggle').addEventListener('click', () => {
  const on = document.documentElement.classList.toggle('nav-collapsed');
  try { localStorage.setItem('nav-collapsed', on ? '1' : '0'); } catch (e) { /* private mode */ }
});

function setPill(id, state, text, title) {
  const pill = $(id);
  pill.className = `status-pill ${state}`;
  $(`${id}-text`).textContent = text;
  if (title) pill.title = title;
}

async function refreshStatus() {
  try {
    const s = await api('/api/status');
    setPill('#st-index', s.chunks ? 'ok' : 'bad', s.chunks ? `база: ${fmt(s.chunks)} фрагм.` : 'база пуста',
      s.chunks ? 'Сколько фрагментов документации в базе' : 'Нажмите «Обновить базу» на странице «Документы»');
    const o = s.ollama;
    if (!o.running) setPill('#st-llm', 'bad', 'Ollama не запущена', `Нет ответа от ${o.url}: запустите приложение Ollama`);
    else if (!o.model_pulled) setPill('#st-llm', 'warn', `нет модели ${o.model}`, `Скачайте модель: ollama pull ${o.model}`);
    else setPill('#st-llm', 'ok', o.model, `Ollama на ${o.url}`);
  } catch (e) {
    setPill('#st-index', 'bad', 'сервер недоступен');
  }
}

// --- shared rendering ------------------------------------------------------------------------

function hitItem(hit, citedNumber) {
  const li = document.createElement('li');
  li.className = `hit${citedNumber ? ' cited' : ''}`;
  li.dataset.id = hit.id;
  li.innerHTML = `
    <div class="hit-head">
      <span class="hit-rank" title="${citedNumber ? `Ответ ссылается на это место как [${citedNumber}]` : 'Место в выдаче поиска'}">${citedNumber || hit.rank}</span>
      <div class="hit-title">${esc(hit.title)}<div class="hit-file">${esc(hit.file)}</div></div>
      <div class="sim" title="Сходство по смыслу с вопросом"><span class="track"><i style="width:${Math.max(0, hit.similarity) * 100}%"></i></span><b>${hit.similarity.toFixed(2)}</b></div>
    </div>
    <div class="hit-text">${esc(hit.text.replace(/^#{1,6} [^\n]*\n+/, ''))}</div>`;
  li.addEventListener('click', () => li.classList.toggle('open'));
  return li;
}

function hbar(label, value, max, text, cls = '') {
  const width = max ? (value / max) * 100 : 0;
  return `<div class="hbar ${cls}"><span class="hb-label">${label}</span>
    <span class="hb-track"><i style="width:${width}%"></i></span><span class="hb-val">${text}</span></div>`;
}

// Minimal Markdown for model answers: code blocks, inline code, bold, lists, paragraphs, [n] markers.
function renderAnswer(text) {
  const blocks = [];
  let src = esc(text).replace(/```[a-z]*\n?([\s\S]*?)(```|$)/g, (_, code) => {
    blocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  src = src.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  src = src.replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-n="$1" title="Показать, откуда это">$1</button>');
  const html = src.split(/\n{2,}/).map((para) => {
    const lines = para.split('\n');
    if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
      const ordered = /^\s*\d+\./.test(lines[0]);
      const items = lines.map((l) => `<li>${l.replace(/^\s*([-*]|\d+\.)\s+/, '')}</li>`).join('');
      return ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
    }
    return /^\u0000\d+\u0000$/.test(para.trim()) ? para : `<p>${lines.join('<br>')}</p>`;
  }).join('');
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

// --- page: ask -------------------------------------------------------------------------------

function initAsk() {
  const form = $('#ask-form'), q = $('#q'), askBtn = $('#ask-btn'), stopBtn = $('#stop-btn');
  let controller = null, timer = null, citeMap = {};

  $('#examples')?.addEventListener('click', (e) => {
    if (e.target.matches('button.chip')) { q.value = e.target.textContent; fit(); form.requestSubmit(); }
  });
  // the question box grows with the text, starting as high as the search box
  const fit = () => { q.style.height = 'auto'; q.style.height = `${q.scrollHeight + 2}px`; };
  q.addEventListener('input', fit);
  q.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  stopBtn.addEventListener('click', () => controller?.abort());

  // Leaving the page closes the stream and stops the model: ask first while an answer is being written.
  const LEAVE_TEXT = 'Ответ ещё пишется. Уйти со страницы и остановить его?';
  let leaving = false;
  document.querySelectorAll('.sidenav a').forEach((link) => link.addEventListener('click', (e) => {
    if (!controller) return;
    if (confirm(LEAVE_TEXT)) leaving = true;
    else e.preventDefault();
  }));
  window.addEventListener('beforeunload', (e) => {
    if (controller && !leaving) { e.preventDefault(); e.returnValue = LEAVE_TEXT; }
  });

  // The last finished answer is kept for this browser tab, so coming back to the page shows it again.
  const SAVED_KEY = 'last-answer';
  function showDone(ev) {
    citeMap = Object.fromEntries(ev.sources.map((s) => [s.number, s.chunk_id]));
    renderHits(ev.results, Object.fromEntries(ev.sources.map((s) => [s.chunk_id, s.number])));
    $('#answer-card').hidden = false;
    $('#answer-text').innerHTML = renderAnswer(ev.answer);
    $('#answer-warn').hidden = !ev.unsupported;
    $('#answer-meta').textContent = `${ev.model} · ${ev.seconds} с`;
    $('#hits-meta').textContent = `ответ ссылается на ${ev.sources.length} из ${ev.results.length}`;
  }

  function renderHits(results, citedById = {}) {
    const list = $('#hits');
    list.innerHTML = '';
    results.forEach((hit) => list.appendChild(hitItem(hit, citedById[hit.id])));
    $('#hits-card').hidden = results.length === 0;
  }

  function setBusy(busy) {
    askBtn.hidden = busy;
    stopBtn.hidden = !busy;
    q.readOnly = busy;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question || controller) return;

    controller = new AbortController();
    setBusy(true);
    citeMap = {};
    $('#answer-card').hidden = false;
    $('#answer-error').hidden = true;
    $('#answer-warn').hidden = true;
    $('#answer-text').innerHTML = '';
    $('#answer-meta').textContent = '';
    $('#hits-card').hidden = true;

    const started = Date.now();
    let text = '', phase = 'Ищу в документации…';
    const tick = () => { $('#wait-text').textContent = `${phase} ${Math.round((Date.now() - started) / 1000)} с`; };
    $('#answer-wait').hidden = false;
    tick();
    timer = setInterval(tick, 500);

    try {
      const response = await fetch('/api/ask/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }), signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === 'results') {
            renderHits(ev.results);
            $('#hits-meta').textContent = `${ev.results.length} мест`;
            phase = 'Модель читает найденное и пишет ответ…';
          } else if (ev.type === 'token') {
            text += ev.text;
            phase = 'Модель пишет ответ…';
            $('#answer-text').innerHTML = renderAnswer(text);
          } else if (ev.type === 'error') {
            throw new Error(ev.detail);
          } else if (ev.type === 'done') {
            showDone(ev);
            try { sessionStorage.setItem(SAVED_KEY, JSON.stringify({ question, ...ev })); } catch (e) { /* storage full or disabled */ }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        $('#answer-meta').textContent = `остановлено через ${Math.round((Date.now() - started) / 1000)} с`;
        if (!text) $('#answer-text').innerHTML = '<p class="muted">Ответ остановлен до начала.</p>';
      } else {
        $('#answer-text').innerHTML = '';  // a derailed reply is garbage: do not leave it on screen
        $('#answer-error').textContent = err.message;
        $('#answer-error').hidden = false;
        refreshStatus();
      }
    } finally {
      clearInterval(timer);
      $('#answer-wait').hidden = true;
      controller = null;
      setBusy(false);
    }
  });

  try {
    const saved = JSON.parse(sessionStorage.getItem(SAVED_KEY) || 'null');
    if (saved) { q.value = saved.question; fit(); showDone(saved); }
  } catch (e) { /* nothing saved */ }

  // [n] in the answer -> scroll to the cited place and open it
  $('#answer-text').addEventListener('click', (e) => {
    const cite = e.target.closest('.cite');
    const id = cite && citeMap[cite.dataset.n];
    const hit = id && document.querySelector(`.hit[data-id="${CSS.escape(id)}"]`);
    if (!hit) return;
    hit.classList.add('open');
    hit.scrollIntoView({ behavior: 'smooth', block: 'center' });
    hit.classList.remove('flash'); void hit.offsetWidth; hit.classList.add('flash');
  });
}

// --- page: search ----------------------------------------------------------------------------

function initSearch() {
  const params = new URLSearchParams(location.search);
  $('#search-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = $('#sq').value.trim();
    if (!query) return;
    history.replaceState(null, '', `?q=${encodeURIComponent(query)}`);
    $('#s-card').hidden = false;
    try {
      const r = await api('/api/search', { query, top_k: 10 });
      const list = $('#s-hits');
      list.innerHTML = '';
      r.results.forEach((hit) => list.appendChild(hitItem(hit)));
      const files = new Set(r.results.map((h) => h.file)).size;
      $('#s-meta').textContent = `${r.results.length} мест из ${files} документов · ${r.ms} мс`;
    } catch (err) {
      $('#s-hits').innerHTML = `<li class="alert bad">${esc(err.message)}</li>`;
    }
  });
  if (params.get('q')) { $('#sq').value = params.get('q'); $('#search-form').requestSubmit(); }
}

// --- page: quality ---------------------------------------------------------------------------

function rankBadge(rank, depth) {
  if (rank === null) return `<span class="rank rx" title="Не найден среди первых ${depth}">нет</span>`;
  const cls = rank === 1 ? 'r1' : rank <= 3 ? 'r3' : 'rx';
  return `<span class="rank ${cls}">${rank}</span>`;
}

async function initQuality() {
  api('/api/experiments').then((x) => {
    const max = Math.max(...x.setups.flatMap((s) => [s.dev.mrr, s.test.mrr]));
    $('#experiments').innerHTML = x.setups.map((s) => `
      <div class="exp${s.current ? ' current' : ''}">
        <div class="exp-label">${esc(s.label)}</div>
        ${hbar('dev', s.dev.mrr, max, s.dev.mrr.toFixed(3))}
        ${hbar('test', s.test.mrr, max, s.test.mrr.toFixed(3), 'test')}
      </div>`).join('');
  }).catch((err) => { $('#experiments').innerHTML = `<p class="muted small">${esc(err.message)}</p>`; });

  let data;
  try { data = await api('/api/eval'); } catch (err) {
    $('#q-table tbody').innerHTML = `<tr><td colspan="4" class="muted">${esc(err.message)}</td></tr>`;
    return;
  }
  const { dev, test } = data.splits;
  $('#t-test-h1').textContent = pct(test.hit_at['1']);
  $('#t-test-h3').textContent = pct(test.hit_at['3']);
  $('#t-test-mrr').textContent = test.mrr.toFixed(2);
  $('#t-dev-mrr').textContent = dev.mrr.toFixed(2);

  const buckets = [['1-е место', (r) => r === 1], ['2–3', (r) => r >= 2 && r <= 3], ['4–5', (r) => r >= 4 && r <= 5],
    ['дальше 5-го', (r) => r > 5], ['не найден', (r) => r === null]];
  const count = (qs, f) => qs.filter((x) => f(x.rank)).length;
  const max = Math.max(...buckets.flatMap(([, f]) => [count(dev.questions, f), count(test.questions, f)]), 1);
  $('#ranks').innerHTML = buckets.map(([label, f]) => {
    const d = count(dev.questions, f), t = count(test.questions, f);
    return `<div class="hbar"><span class="hb-label">${label}</span>
      <span class="hb-track double"><i style="width:${(d / max) * 100}%"></i><i style="width:${(t / max) * 100}%"></i></span>
      <span class="hb-val">${d} · ${t}</span></div>`;
  }).join('') + `<p class="muted small">${dev.questions.length} вопросов в dev и ${test.questions.length} в test.</p>`;

  const rows = [];
  for (const [split, s] of Object.entries(data.splits)) {
    s.questions.forEach((x) => rows.push({ split, ...x, neg: false }));
    s.negatives.forEach((question) => rows.push({ split, question, neg: true }));
  }
  const tbody = $('#q-table tbody');
  const draw = (filter) => {
    const shown = rows.filter((r) => filter === 'all' || r.split === filter
      || (filter === 'miss' && !r.neg && r.rank !== 1) || (filter === 'neg' && r.neg));
    tbody.innerHTML = shown.map((r) => r.neg
      ? `<tr><td><span class="rank neg">—</span></td><td>${esc(r.question)}<span class="split-tag">${r.split}</span></td>
           <td class="muted" colspan="2">ответа в документах нет — модель должна отказаться</td></tr>`
      : `<tr><td>${rankBadge(r.rank, data.depth)}</td>
           <td><a href="/search?q=${encodeURIComponent(r.question)}" title="Открыть в поиске">${esc(r.question)}</a><span class="split-tag">${r.split}</span></td>
           <td class="mono">${r.expected.map(esc).join('<br>')}</td>
           <td class="mono${r.rank === 1 ? ' muted' : ''}">${esc(r.top_source)}</td></tr>`).join('');
  };
  $('#q-filter').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    for (const x of $('#q-filter').children) x.classList.toggle('on', x === b);
    draw(b.dataset.f);
  });
  draw('all');
}

// --- page: documents -------------------------------------------------------------------------

async function initDocuments() {
  const load = async () => {
    const c = await api('/api/documents');
    $('#c-docs').textContent = c.documents;
    $('#c-chunks').textContent = fmt(c.chunks);
    $('#c-chars').textContent = fmt(c.chars);
    const files = [...c.files].sort((a, b) => a.title.localeCompare(b.title, 'ru'));
    const draw = (needle) => {
      const n = needle.toLowerCase();
      const shown = files.filter((f) => !n || f.file.toLowerCase().includes(n) || f.title.toLowerCase().includes(n));
      $('#docs tbody').innerHTML = shown.map((f) => `<tr><td>${esc(f.title)}</td><td class="mono muted">${esc(f.file)}</td>
        <td class="num">${f.chunks}</td><td class="num">${fmt(f.chars)}</td></tr>`).join('')
        || '<tr><td colspan="4" class="muted">Ничего не найдено.</td></tr>';
    };
    $('#doc-filter').oninput = (e) => draw(e.target.value.trim());
    draw($('#doc-filter').value.trim());
  };
  await load();

  $('#reindex').addEventListener('click', async (e) => {
    const btn = e.target;
    btn.disabled = true;
    $('#reindex-meta').textContent = 'Обновляю…';
    try {
      const r = await api('/api/index', {});
      $('#reindex-meta').textContent = r.added || r.updated || r.removed
        ? `добавлено ${r.added}, изменено ${r.updated}, удалено ${r.removed} фрагментов · ${r.seconds} с`
        : `изменений нет · ${r.seconds} с`;
      await load();
      refreshStatus();
    } catch (err) {
      $('#reindex-meta').textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  });
}

// --- boot ------------------------------------------------------------------------------------

refreshStatus();
setInterval(refreshStatus, 15000);
({ ask: initAsk, search: initSearch, quality: initQuality, documents: initDocuments })[document.body.dataset.page]?.();
