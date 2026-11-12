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
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
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
    setPill('#st-index', s.chunks ? 'ok' : 'bad', s.chunks ? `${fmt(s.chunks)} чанков` : 'индекс пуст',
      s.chunks ? 'Чанков в векторной базе' : 'Запустите python main.py index или «Переиндексировать» на странице «Документы»');
    const o = s.ollama;
    if (!o.running) setPill('#st-llm', 'bad', 'Ollama не запущена', `Нет ответа от ${o.url}: запустите приложение Ollama или ollama serve`);
    else if (!o.model_pulled) setPill('#st-llm', 'warn', `нет ${o.model}`, `Модель не скачана: ollama pull ${o.model}`);
    else setPill('#st-llm', 'ok', o.model, `Ollama на ${o.url}`);
    return s;
  } catch (e) {
    setPill('#st-index', 'bad', 'сервер недоступен');
    return null;
  }
}

// --- shared rendering ------------------------------------------------------------------------

function hitItem(hit, citedNumber) {
  const li = document.createElement('li');
  li.className = `hit${citedNumber ? ' cited' : ''}`;
  li.dataset.id = hit.id;
  li.style.animationDelay = `${(hit.rank - 1) * 40}ms`;
  li.title = 'Клик — показать текст фрагмента целиком';
  li.innerHTML = `
    <div class="hit-head">
      <span class="hit-rank">${citedNumber || hit.rank}</span>
      <div class="hit-title">${esc(hit.title)}<div class="hit-file">${esc(hit.file)}</div></div>
      <div class="sim"><span class="track"><i style="width:${Math.max(0, hit.similarity) * 100}%"></i></span><b>${hit.similarity.toFixed(3)}</b></div>
    </div>
    <div class="hit-text">${esc(hit.text)}</div>`;
  li.addEventListener('click', () => li.classList.toggle('open'));
  return li;
}

function hbar(label, value, max, text, cls = '') {
  const width = max ? (value / max) * 100 : 0;
  return `<div class="hbar ${cls}"><span class="hb-label" title="${esc(label)}">${label}</span>
    <span class="hb-track"><i style="width:${width}%"></i></span><span class="hb-val">${text}</span></div>`;
}

// Minimal Markdown for model answers: code blocks, inline code, bold, lists, paragraphs, [n] markers.
function renderAnswer(text) {
  const blocks = [];
  let src = esc(text).replace(/```[a-z]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  src = src.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  src = src.replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-n="$1" title="Показать источник [$1]">$1</button>');
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
  const form = $('#ask-form'), q = $('#q'), btn = $('#ask-btn');
  let timer = null;

  $('#examples').addEventListener('click', (e) => {
    if (e.target.matches('button.chip')) { q.value = e.target.textContent; form.requestSubmit(); }
  });
  q.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  function show(id) {
    for (const part of ['#answer-empty', '#answer-wait', '#answer-error', '#answer-body']) $(part).hidden = part !== id;
  }

  function renderHits(results, citedById = {}) {
    const list = $('#hits');
    list.innerHTML = '';
    $('#hits-empty').hidden = results.length > 0;
    results.forEach((hit) => list.appendChild(hitItem(hit, citedById[hit.id])));
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question || btn.disabled) return;
    const topK = Number($('#top-k').value);
    btn.disabled = true;
    show('#answer-wait');
    $('#answer-meta').textContent = '';

    // Chunks first: retrieval takes milliseconds, the model takes about a minute on CPU.
    try {
      const found = await api('/api/search', { query: question, top_k: topK });
      renderHits(found.results);
      $('#hits-meta').textContent = `${found.results.length} фрагментов · ${found.ms} мс`;
    } catch (err) { /* the answer request below reports the problem */ }

    const started = Date.now();
    const tick = () => { $('#wait-text').textContent = `Модель читает фрагменты и пишет ответ… ${Math.round((Date.now() - started) / 1000)} с`; };
    tick();
    timer = setInterval(tick, 500);
    try {
      const a = await api('/api/ask', { question, top_k: topK });
      const citedById = Object.fromEntries(a.sources.map((s) => [s.chunk_id, s.number]));
      renderHits(a.results, citedById);
      $('#answer-text').innerHTML = renderAnswer(a.answer);
      $('#answer-warn').hidden = !a.unsupported;
      $('#sources').innerHTML = a.sources.length
        ? a.sources.map((s) => `<li><button type="button" class="cite" data-n="${s.number}">${s.number}</button>
            <span>${esc(s.title)} <span class="file">${esc(s.file)}</span></span></li>`).join('')
        : '<li class="muted">Модель не сослалась ни на один фрагмент.</li>';
      $('#answer-body').dataset.map = JSON.stringify(Object.fromEntries(a.sources.map((s) => [s.number, s.chunk_id])));
      $('#answer-meta').textContent = `${a.model} · ${a.seconds} с · процитировано ${a.sources.length} из ${a.results.length}`;
      show('#answer-body');
    } catch (err) {
      $('#answer-error').textContent = err.message;
      show('#answer-error');
      refreshStatus();
    } finally {
      clearInterval(timer);
      btn.disabled = false;
    }
  });

  // [n] in the answer or in the source list -> scroll to the cited chunk and open it
  $('#answer-body').addEventListener('click', (e) => {
    const cite = e.target.closest('.cite');
    if (!cite) return;
    const id = JSON.parse($('#answer-body').dataset.map || '{}')[cite.dataset.n];
    const hit = id && document.querySelector(`.hit[data-id="${CSS.escape(id)}"]`);
    if (!hit) return;
    hit.classList.add('open');
    hit.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    try {
      const r = await api('/api/search', { query, top_k: Number($('#s-top-k').value) });
      const list = $('#s-hits');
      list.innerHTML = '';
      r.results.forEach((hit) => list.appendChild(hitItem(hit)));
      $('#s-empty').hidden = r.results.length > 0;
      const files = new Set(r.results.map((h) => h.file));
      $('#t-ms').textContent = r.ms;
      $('#t-top').textContent = r.results.length ? r.results[0].similarity.toFixed(3) : '–';
      $('#t-files').textContent = files.size;
      $('#t-best').textContent = r.results.length ? r.results[0].file : '–';
      $('#t-best').title = $('#t-best').textContent;
      $('#s-tiles').hidden = false;
    } catch (err) {
      $('#s-empty').hidden = false;
      $('#s-empty').textContent = err.message;
    }
  });
  if (params.get('q')) { $('#sq').value = params.get('q'); $('#search-form').requestSubmit(); }
}

// --- page: quality ---------------------------------------------------------------------------

function rankBadge(rank, depth) {
  if (rank === null) return `<span class="rank rx" title="Не найден в top-${depth}">>${depth}</span>`;
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
      </div>`).join('') +
      '<p class="muted small">Модель эмбеддингов — главный фактор. Различия меньше ~0.04 MRR — это один вопрос из 24, т. е. шум.</p>';
  }).catch((err) => { $('#experiments').innerHTML = `<p class="muted small">${esc(err.message)}</p>`; });

  let data;
  try { data = await api('/api/eval'); } catch (err) {
    $('#q-table tbody').innerHTML = `<tr><td colspan="4" class="muted">${esc(err.message)}</td></tr>`;
    return;
  }
  const { dev, test } = data.splits;
  $('#t-dev-mrr').textContent = dev.mrr.toFixed(3);
  $('#t-test-mrr').textContent = test.mrr.toFixed(3);
  $('#t-test-h1').textContent = pct(test.hit_at['1']);
  $('#t-test-h3').textContent = pct(test.hit_at['3']);
  $('#t-count').textContent = dev.questions.length + test.questions.length + dev.negatives.length + test.negatives.length;

  // rank distribution, dev and test bars side by side
  const buckets = [['1-е место', (r) => r === 1], ['2–3', (r) => r >= 2 && r <= 3], ['4–5', (r) => r >= 4 && r <= 5],
    ['6–20', (r) => r > 5], ['не найден', (r) => r === null]];
  const count = (qs, f) => qs.filter((x) => f(x.rank)).length;
  const max = Math.max(...buckets.flatMap(([, f]) => [count(dev.questions, f), count(test.questions, f)]), 1);
  $('#ranks').innerHTML = buckets.map(([label, f]) => {
    const d = count(dev.questions, f), t = count(test.questions, f);
    return `<div class="hbar"><span class="hb-label">${label}</span>
      <span class="hb-track double"><i style="width:${(d / max) * 100}%"></i><i style="width:${(t / max) * 100}%"></i></span>
      <span class="hb-val">${d} · ${t}</span></div>`;
  }).join('') + `<p class="muted small">Из ${dev.questions.length} dev и ${test.questions.length} test вопросов; глубина поиска ${data.depth}.</p>`;

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
           <td class="muted" colspan="2">ответа в документации нет — проверка отказа модели</td></tr>`
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
    $('#c-avg').textContent = fmt(c.avg_chunk);
    $('#c-chars').textContent = fmt(c.chars);
    const hmax = Math.max(...c.histogram.map((h) => h.count), 1);
    $('#histogram').innerHTML = c.histogram.map((h) => hbar(h.label, h.count, hmax, h.count)).join('')
      + '<p class="muted small">Раздел длиннее 1000 символов режется на части, поэтому чанков длиннее ~1000 почти нет.</p>';
    const fmax = Math.max(...c.files.map((f) => f.chunks), 1);
    const draw = (needle) => {
      const n = needle.toLowerCase();
      $('#docs').innerHTML = c.files.filter((f) => !n || f.file.includes(n) || f.title.toLowerCase().includes(n))
        .map((f) => hbar(`${esc(f.title)} <span class="mono">${esc(f.file)}</span>`, f.chunks, fmax, f.chunks)).join('')
        || '<p class="muted small">Ничего не найдено.</p>';
    };
    $('#doc-filter').oninput = (e) => draw(e.target.value.trim());
    draw($('#doc-filter').value.trim());
  };
  await load();

  $('#reindex').addEventListener('click', async (e) => {
    const btn = e.target;
    btn.disabled = true;
    $('#reindex-meta').textContent = 'Синхронизация…';
    try {
      const r = await api('/api/index', {});
      $('#reindex-meta').textContent = `+${r.added} новых · ${r.updated} изменено · ${r.removed} удалено · `
        + `${r.unchanged} без изменений · ${r.seconds} с`;
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
