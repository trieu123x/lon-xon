/* CNN Lab — Assignment 05.
 * Toàn bộ suy luận chạy trong trình duyệt bằng ONNX Runtime Web (WASM).
 * Mô hình và mô tả (models/*.json) là file tĩnh; không có API phía máy chủ.
 */
'use strict';

const ORT_VERSION = '1.30.0';
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// Đa luồng WASM cần trang được cô lập cross-origin (COOP/COEP); nếu không thì chạy 1 luồng.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const fmtPct = (p, d = 1) => `${(100 * p).toFixed(d)}%`;
const fmtParams = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} tr` : `${Math.round(n / 1e3)} K`) + ' tham số';

/* ---------------------------------------------------------------- tải dữ liệu & mô hình */
const metaCache = new Map();
function loadMeta(ds) {
  if (!metaCache.has(ds)) {
    metaCache.set(ds, fetch(`models/${ds}.json`).then((r) => {
      if (!r.ok) throw new Error(`Không tải được models/${ds}.json (${r.status})`);
      return r.json();
    }));
  }
  return metaCache.get(ds);
}

async function fetchBytes(url, expectedBytes, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Không tải được ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length')) || expectedBytes || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(total ? Math.min(got / total, 1) : 0, got);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

const sessionCache = new Map();
function getSession(model, onProgress) {
  if (!sessionCache.has(model.file)) {
    const p = fetchBytes(model.file, (model.size_mb || 0) * 1e6, onProgress)
      .then((bytes) => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }));
    p.catch(() => sessionCache.delete(model.file));
    sessionCache.set(model.file, p);
  }
  return sessionCache.get(model.file);
}

function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function runModel(session, data, dims) {
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', data, dims) };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  return { probs: softmax(Array.from(out[session.outputNames[0]].data)), ms };
}

/* ---------------------------------------------------------------- tiền xử lý ảnh
 * Giống hệt Python: cắt hình vuông ở giữa (tỉ lệ `keep` của cạnh ngắn), thu về N×N, chia 255.
 *   CIFAR-10: keep = 1       (ảnh vuông -> 32×32)
 *   Flowers : keep = 128/144 (vuông -> 144 -> cắt tâm 128  ≡  cắt tâm 128/144 -> 128)
 * Khi thu nhỏ nhiều lần, ta chia đôi dần từng bước để tránh răng cưa (gần với
 * phép lọc chống răng cưa của PIL). Với ảnh mẫu 32×32 / 144×144, phép vẽ không
 * đổi kích thước nên điểm ảnh được giữ nguyên tuyệt đối.
 */
function preprocess(img, size, keep) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const side = Math.min(w, h) * keep;
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  let cur = Math.round(side);
  let src = document.createElement('canvas');
  src.width = src.height = cur;
  let ctx = src.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, sx, sy, side, side, 0, 0, cur, cur);
  while (cur / 2 >= size * 1.5) {
    const next = Math.round(cur / 2);
    const c = document.createElement('canvas');
    c.width = c.height = next;
    const cx = c.getContext('2d');
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(src, 0, 0, cur, cur, 0, 0, next, next);
    src = c; cur = next;
  }
  const out = document.createElement('canvas');
  out.width = out.height = size;
  ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, cur, cur, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = px[i * 4] / 255;
    data[n + i] = px[i * 4 + 1] / 255;
    data[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return { data, canvas: out };
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Không đọc được ảnh'));
    img.src = src;
  });
}

/* ---------------------------------------------------------------- bảng ảnh */
const IMAGE_DS = {
  cifar10: {
    size: 32, keep: 1, topk: 5,
    intro: '<p><b>60.000 ảnh 32×32, 10 lớp</b> (Kaggle <code>quanbk/cifar10</code>). Bốn kiến trúc ở Phần 2 được '
      + 'huấn luyện <b>từ đầu</b> với cùng một công thức: 20 epoch, AdamW, OneCycle, tăng cường dữ liệu bằng cắt ngẫu nhiên và lật ngang. '
      + 'Ảnh tải lên được cắt hình vuông ở giữa rồi thu về <b>32×32</b>, đúng độ phân giải mô hình được huấn luyện. Ảnh chụp '
      + 'thật thường khác xa phân bố CIFAR, nên nên thử ảnh vật thể nằm giữa khung.</p>',
    hint: 'Ảnh sẽ bị thu về 32×32 điểm ảnh',
    inputCap: 'mô hình thấy (32×32)',
  },
  flowers: {
    size: 128, keep: 128 / 144, topk: 5,
    intro: '<p><b>4.317 ảnh, 5 loài hoa</b> (Kaggle <code>alxmamaev/flowers-recognition</code>), sau khi loại ảnh trùng lặp. '
      + 'LeNet-5, AlexNet, VGG-16 được huấn luyện <b>từ đầu</b>; ResNet-50 được <b>học chuyển giao</b> từ ImageNet. Với chỉ '
      + 'khoảng 3.000 ảnh train, đây là khác biệt lớn nhất trong thí nghiệm, xem notebook Bài 2.</p>',
    hint: 'Nên dùng ảnh hoa chụp gần, bông hoa ở giữa khung',
    inputCap: 'mô hình thấy (128×128)',
  },
};

function makeImagePanel(ds) {
  const cfg = IMAGE_DS[ds];
  const host = $(`#tab-${ds}`);
  host.append($('#imagePanelTpl').content.cloneNode(true));
  const root = host.firstElementChild;
  root.id = `panel-${ds}`;
  $('.intro', root).innerHTML = cfg.intro;
  $('.drop-hint', root).textContent = cfg.hint;

  const state = { meta: null, modelId: null, img: null, label: null, busy: false, token: 0 };
  const els = {
    samples: $('.samples', root), models: $('.models', root), placeholder: $('.placeholder', root),
    body: $('.res-body', root), orig: $('canvas.orig', root), input: $('canvas.input', root),
    verdict: $('.verdict', root), bars: $('.bars', root), compare: $('.compare', root), meta: $('.meta', root),
    progress: $('.progress', root), pbar: $('.progress-bar', root), ptxt: $('.progress-txt', root),
    file: $('input[type=file]', root), drop: $('.drop', root), cmpBtn: $('.compare-btn', root),
  };
  $('.input-cap', root).textContent = cfg.inputCap;

  const classLabel = (k) => `${state.meta.classes_vi[k]} <span class="muted">(${state.meta.classes[k]})</span>`;
  const progress = (name) => (frac, got) => {
    els.progress.hidden = false;
    els.pbar.style.width = `${(frac * 100).toFixed(1)}%`;
    els.ptxt.textContent = `Đang tải ${name}: ${(got / 1e6).toFixed(1)} MB${frac ? ` (${Math.round(frac * 100)}%)` : ''}`;
  };

  async function init() {
    state.meta = await loadMeta(ds);
    const best = state.meta.models.reduce((a, b) => (b.test_acc > a.test_acc ? b : a));
    state.modelId = best.id;
    els.models.innerHTML = state.meta.models.map((m) => `
      <label class="model">
        <input type="radio" name="model-${ds}" value="${m.id}" ${m.id === best.id ? 'checked' : ''}>
        <b>${m.name}${m.id === best.id ? '<span class="tag">tốt nhất</span>' : ''}</b>
        <span class="acc">${fmtPct(m.test_acc)}</span>
        <small>${fmtParams(m.params)} · ${m.size_mb.toFixed(m.size_mb < 1 ? 2 : 1)} MB${m.pretrained ? ' · khởi tạo ImageNet' : ''}</small>
      </label>`).join('');
    els.models.addEventListener('change', (e) => { state.modelId = e.target.value; predict(); });

    els.samples.innerHTML = state.meta.samples.map((s, i) => `
      <button type="button" data-i="${i}" data-ds="${ds}" title="${state.meta.classes[s.label]}">
        <img src="${s.file}" alt="${state.meta.classes[s.label]}" loading="lazy">
      </button>`).join('');
    els.samples.addEventListener('click', async (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const s = state.meta.samples[+b.dataset.i];
      $$('button', els.samples).forEach((x) => x.classList.toggle('sel', x === b));
      setImage(await loadImage(s.file), s.label);
    });

    const acc = state.meta.models.filter((m) => m.per_class_acc);
    const pc = $('.perclass', root);
    if (acc.length) {
      pc.innerHTML = `<table><thead><tr><th>Mô hình</th>${state.meta.classes_vi.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>
        ${acc.map((m) => `<tr><td>${m.name}</td>${m.per_class_acc.map((v) => `<td>${fmtPct(v, 1)}</td>`).join('')}</tr>`).join('')}
        </tbody></table>`;
    } else {
      root.querySelector('.extra').remove();
    }
  }

  function setImage(img, label = null) {
    state.img = img;
    state.label = label;
    const o = els.orig;
    o.width = o.height = 224;
    const w = img.naturalWidth, h = img.naturalHeight, side = Math.min(w, h);
    const octx = o.getContext('2d');
    octx.imageSmoothingEnabled = w > 64;
    octx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 224, 224);
    els.compare.innerHTML = '';
    predict();
  }

  async function predict() {
    if (!state.img || !state.meta) return;
    const token = ++state.token;
    const model = state.meta.models.find((m) => m.id === state.modelId);
    const { data, canvas } = preprocess(state.img, cfg.size, cfg.keep);
    els.placeholder.hidden = true;
    els.body.hidden = false;
    const ictx = els.input.getContext('2d');
    els.input.width = els.input.height = cfg.size;
    ictx.drawImage(canvas, 0, 0);
    try {
      const session = await getSession(model, progress(model.name));
      els.progress.hidden = true;
      const { probs, ms } = await runModel(session, data, [1, 3, cfg.size, cfg.size]);
      if (token !== state.token) return;
      renderResult(model, probs, ms);
    } catch (err) {
      els.progress.hidden = true;
      els.verdict.innerHTML = `<span class="pill no">Lỗi</span> ${err.message}`;
    }
  }

  function renderResult(model, probs, ms) {
    const order = probs.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]);
    const [pTop, kTop] = order[0];
    let tag = '';
    if (state.label !== null) {
      tag = kTop === state.label
        ? '<span class="pill ok">đúng</span>'
        : `<span class="pill no">sai — nhãn thật: ${state.meta.classes_vi[state.label]}</span>`;
    }
    els.verdict.innerHTML = `${model.name} đoán: <b>${state.meta.classes_vi[kTop]}</b> · ${fmtPct(pTop)}${tag}`;
    els.bars.innerHTML = order.slice(0, cfg.topk).map(([p, k], i) => `
      <div class="barrow ${i === 0 ? 'top' : ''}">
        <span class="name">${classLabel(k)}</span>
        <span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
        <span class="v">${fmtPct(p)}</span>
      </div>`).join('');
    els.meta.textContent = `Suy luận ${ms.toFixed(0)} ms trên CPU (WASM, ${ort.env.wasm.numThreads} luồng) · file ${model.size_mb} MB · accuracy test ${fmtPct(model.test_acc, 2)}`;
  }

  async function compareAll() {
    if (!state.img) { els.placeholder.textContent = 'Hãy chọn ảnh trước.'; return; }
    els.cmpBtn.disabled = true;
    const { data } = preprocess(state.img, cfg.size, cfg.keep);
    const rows = [];
    try {
      for (const m of state.meta.models) {
        const session = await getSession(m, progress(m.name));
        const { probs, ms } = await runModel(session, data, [1, 3, cfg.size, cfg.size]);
        const k = probs.indexOf(Math.max(...probs));
        rows.push({ m, k, p: probs[k], ms, pTrue: state.label !== null ? probs[state.label] : null });
      }
    } catch (err) {
      els.compare.innerHTML = `<p class="pill no">${err.message}</p>`;
    }
    els.progress.hidden = true;
    els.cmpBtn.disabled = false;
    const best = rows.reduce((a, b) => (b.p > a.p ? b : a), rows[0]);
    els.compare.innerHTML = `<table class="cmp"><thead><tr><th>Mô hình</th><th>Dự đoán</th><th>Độ tin</th>
      ${state.label !== null ? '<th>P(nhãn thật)</th>' : ''}<th>ms</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="${r === best ? 'win' : ''}"><td>${r.m.name}</td>
        <td>${state.meta.classes_vi[r.k]} ${state.label !== null ? (r.k === state.label ? '✓' : '✗') : ''}</td>
        <td class="num">${fmtPct(r.p)}</td>
        ${state.label !== null ? `<td class="num">${fmtPct(r.pTrue)}</td>` : ''}
        <td class="num">${r.ms.toFixed(0)}</td></tr>`).join('')}</tbody></table>`;
  }

  // nhập ảnh: chọn file, kéo-thả, dán
  const readFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return;
    const url = URL.createObjectURL(f);
    loadImage(url).then((img) => { $$('button', els.samples).forEach((x) => x.classList.remove('sel')); setImage(img, null); });
  };
  els.file.addEventListener('change', () => readFile(els.file.files[0]));
  els.drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.file.click(); } });
  ['dragenter', 'dragover'].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove('over'); }));
  els.drop.addEventListener('drop', (e) => readFile(e.dataTransfer.files[0]));
  document.addEventListener('paste', (e) => {
    if (!root.closest('.tabpane').classList.contains('active')) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) readFile(item.getAsFile());
  });
  els.cmpBtn.addEventListener('click', compareAll);

  return { init };
}

/* ---------------------------------------------------------------- diabetes */
function makeDiabetes() {
  const form = $('#dbForm');
  const out = $('#dbOut');
  let meta = null;

  function values() {
    const f = new FormData(form);
    return {
      gender: f.get('gender'), age: +f.get('age'), bmi: +f.get('bmi'),
      HbA1c_level: +f.get('HbA1c_level'), blood_glucose_level: +f.get('blood_glucose_level'),
      smoking_history: f.get('smoking_history'),
      hypertension: f.get('hypertension') ? 1 : 0, heart_disease: f.get('heart_disease') ? 1 : 0,
    };
  }

  // phải khớp hàm encode() trong notebook Bài 3
  function encode(v) {
    const base = [v.age, v.bmi, v.HbA1c_level, v.blood_glucose_level, v.hypertension, v.heart_disease,
      v.gender === 'Male' ? 1 : 0];
    return Float32Array.from([...base, ...meta.smoking.map((s) => (v.smoking_history === s ? 1 : 0))]);
  }

  function syncOutputs() {
    $$('output', form).forEach((o) => { o.textContent = form.elements[o.dataset.for].value; });
  }

  function fill(s) {
    for (const k of ['gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']) {
      form.elements[k].value = s[k];
    }
    form.elements.hypertension.checked = !!s.hypertension;
    form.elements.heart_disease.checked = !!s.heart_disease;
    syncOutputs();
  }

  async function predict() {
    syncOutputs();
    const v = values();
    const x = encode(v);
    const res = [];
    for (const m of meta.models) {
      const s = await getSession(m);
      const { probs, ms } = await runModel(s, x, [1, x.length]);
      res.push({ m, p: probs[1], ms });
    }
    const rule = v.HbA1c_level > meta.rule.hba1c_gt || v.blood_glucose_level > meta.rule.glucose_gt;
    out.innerHTML = `<div class="prob-grid">${res.map(({ m, p, ms }) => `
      <div class="gauge ${p >= m.threshold ? 'pos' : 'neg'}">
        <div class="lbl">${m.name} · P(tiểu đường)</div>
        <div class="p">${fmtPct(p)}</div>
        <div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div>
          <div class="thr" style="left:${(m.threshold * 100).toFixed(1)}%" title="ngưỡng"></div></div>
        <div class="lbl">${p >= m.threshold ? 'Nguy cơ <b>cao</b>' : 'Nguy cơ <b>thấp</b>'} (ngưỡng F1 tối ưu ${m.threshold.toFixed(2)}) · ${ms.toFixed(1)} ms</div>
        <div class="lbl">test: ROC-AUC ${m.roc_auc.toFixed(4)} · F1 ${m.f1.toFixed(3)}</div>
      </div>`).join('')}</div>
      ${rule
        ? `<div class="alert warn">HbA1c &gt; ${meta.rule.hba1c_gt}% hoặc đường huyết &gt; ${meta.rule.glucose_gt} mg/dL: trong toàn bộ 100.000 hồ sơ,
           <b>100%</b> người vượt ngưỡng này đều có nhãn tiểu đường (trùng tiêu chuẩn chẩn đoán ADA).</div>`
        : `<div class="alert info">Hồ sơ nằm trong <b>"vùng xám"</b> (chưa vượt ngưỡng HbA1c/đường huyết). Ở đây mô hình phải dựa vào tuổi, BMI,
           tăng huyết áp… và đây là phần khó nhất của bài toán: chỉ khoảng 3% hồ sơ vùng xám có nhãn dương.</div>`}`;
  }

  async function init() {
    meta = await loadMeta('diabetes');
    const labels = ['Dương · rõ', 'Dương · rõ', 'Âm', 'Âm', 'Dương · vùng xám', 'Dương · vùng xám'];
    $('#dbSamples').innerHTML = meta.samples.map((s, i) => `<button type="button" data-i="${i}">
      ${labels[i] || `Mẫu ${i + 1}`} · ${s.gender === 'Male' ? 'nam' : 'nữ'} ${Math.round(s.age)}t</button>`).join('');
    $('#dbSamples').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      fill(meta.samples[+b.dataset.i]);
      predict();
    });
    form.addEventListener('input', predict);
    syncOutputs();
    await predict();
  }
  return { init };
}

/* ---------------------------------------------------------------- tự kiểm chứng parity (cũng được test tự động gọi) */
async function selfTest(log = () => {}) {
  const report = { image: [], diabetes: [], maxDiff: 0, argmaxMismatch: 0, n: 0 };
  for (const ds of ['cifar10', 'flowers']) {
    const meta = await loadMeta(ds);
    const cfg = IMAGE_DS[ds];
    const imgs = await Promise.all(meta.samples.map((s) => loadImage(s.file)));
    for (const m of meta.models) {
      const session = await getSession(m);
      let maxd = 0, mism = 0;
      for (let i = 0; i < imgs.length; i++) {
        const { data } = preprocess(imgs[i], cfg.size, cfg.keep);
        const { probs } = await runModel(session, data, [1, 3, cfg.size, cfg.size]);
        const ref = meta.samples[i].ref[m.id];
        probs.forEach((p, k) => { maxd = Math.max(maxd, Math.abs(p - ref[k])); });
        if (probs.indexOf(Math.max(...probs)) !== ref.indexOf(Math.max(...ref))) mism++;
      }
      report.image.push({ ds, model: m.id, n: imgs.length, maxDiff: maxd, argmaxMismatch: mism });
      report.maxDiff = Math.max(report.maxDiff, maxd);
      report.argmaxMismatch += mism;
      report.n += imgs.length;
      log(`${ds.padEnd(8)} ${m.name.padEnd(22)} ${imgs.length} ảnh  max|Δp| = ${maxd.toExponential(2)}  lệch nhãn = ${mism}`);
    }
  }
  const meta = await loadMeta('diabetes');
  for (const m of meta.models) {
    const session = await getSession(m);
    let maxd = 0;
    for (const s of meta.samples) {
      const x = Float32Array.from([s.age, s.bmi, s.HbA1c_level, s.blood_glucose_level, s.hypertension, s.heart_disease,
        s.gender === 'Male' ? 1 : 0, ...meta.smoking.map((k) => (s.smoking_history === k ? 1 : 0))]);
      const { probs } = await runModel(session, x, [1, x.length]);
      maxd = Math.max(maxd, Math.abs(probs[1] - s.ref[m.id]));
    }
    report.diabetes.push({ model: m.id, n: meta.samples.length, maxDiff: maxd });
    report.maxDiff = Math.max(report.maxDiff, maxd);
    report.n += meta.samples.length;
    log(`diabetes ${m.name.padEnd(22)} ${meta.samples.length} hồ sơ max|Δp| = ${maxd.toExponential(2)}`);
  }
  log(`\nTổng: ${report.n} phép so · max|Δp| = ${report.maxDiff.toExponential(2)} · lệch nhãn = ${report.argmaxMismatch}`);
  return report;
}
window.__selfTest = selfTest;

/* ---------------------------------------------------------------- khởi tạo theo trang
 * Mỗi trang (index / cifar10 / flowers / diabetes) là một file HTML tĩnh riêng,
 * khai báo <body data-page="...">; app.js chỉ dựng phần giao diện của trang đó.
 */
const PAGES = {
  cifar10: () => makeImagePanel('cifar10'),
  flowers: () => makeImagePanel('flowers'),
  diabetes: () => makeDiabetes(),
};
const page = document.body.dataset.page;
if (PAGES[page]) {
  PAGES[page]().init().catch((err) => {
    $('main').insertAdjacentHTML('afterbegin', `<p class="alert warn">${err.message}</p>`);
  });
}

$('#ortVer').textContent = ORT_VERSION;
$('#themeBtn').addEventListener('click', () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('theme', document.documentElement.dataset.theme); } catch (_) { /* không sao */ }
});
try {
  const t = localStorage.getItem('theme');
  if (t) document.documentElement.dataset.theme = t;
} catch (_) { /* chế độ riêng tư */ }

$('#selfTestBtn')?.addEventListener('click', async (e) => {
  const pre = $('#selfTestOut');
  pre.hidden = false;
  pre.textContent = '';
  e.target.disabled = true;
  try {
    await selfTest((line) => { pre.textContent += `${line}\n`; });
  } catch (err) {
    pre.textContent += `Lỗi: ${err.message}`;
  }
  e.target.disabled = false;
});
