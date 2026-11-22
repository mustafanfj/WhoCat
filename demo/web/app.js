/* WhoCat demo: every decision is computed in the browser from 512-d face embeddings.
   Sample homes come from bundle/bundle.json (demo/export_bundle.py); with demo/server.py running,
   your own photos are embedded by the real models through /api/embed. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const LETTERS = "ABCDEFGH";

  const S = {
    bundle: null, live: false, home: 0, model: "ours", rule: "full", cam: "clean",
    enrol: [], memory: [], memOn: true, visitor: null, candidates: [], uploads: [], log: null,
  };
  const embCache = new Map();

  // ---------- maths ----------
  function decode(b64) {
    const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(buf);
  }
  function emb(src, model, variant) {
    const key = `${src}|${model}|${variant}`;
    if (!embCache.has(key)) {
      const rec = S.bundle.images[src] || S.uploads.find(u => u.src === src);
      embCache.set(key, decode(rec.emb[model][variant]));
    }
    return embCache.get(key);
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  function template(srcs, model) {
    const d = S.bundle.dim, t = new Float32Array(d);
    srcs.forEach(src => { const e = emb(src, model, "clean"); for (let i = 0; i < d; i++) t[i] += e[i]; });
    const n = Math.hypot(...t) || 1;
    for (let i = 0; i < d; i++) t[i] /= n;
    return t;
  }

  function decide(visitorSrc) {
    const cfg = S.bundle.models[S.model];
    const enrolled = S.enrol.filter(e => e.on).map(e => e.src);
    const k = String(Math.min(3, Math.max(1, enrolled.length)));
    const full = S.rule === "full";
    const t = full ? cfg.t_lookalike[k] : cfg.t_random;
    const m = full ? cfg.margin[k] : null;
    const x = emb(visitorSrc, S.model, S.cam);
    const so = dot(x, template(enrolled, S.model));
    const mem = S.memory.filter(n => n.on);
    const sims = mem.map(n => ({ n, s: dot(x, emb(n.stored, S.model, "clean")) })).sort((a, b) => b.s - a.s);
    const pass1 = so > t;
    const useMem = full && S.memOn && mem.length > 0 && m !== null && m !== undefined;
    let pass2 = null, why2 = "";
    if (!full) why2 = "Not used by the threshold-only rule.";
    else if (!S.memOn || mem.length === 0) why2 = "Neighbour memory is off.";
    else if (m === null || m === undefined) why2 = "No margin was tuned for this model.";
    if (useMem) pass2 = so - sims[0].s > m;
    const open = pass1 && (pass2 === null || pass2);
    return { so, t, m, k, sims, pass1, pass2, useMem, why2, open };
  }

  // ---------- data ----------
  function truthOf(src) {
    const h = S.bundle.homes[S.home];
    if (h.owner_visits.includes(src) || h.enrol.includes(src)) return { kind: "owner", label: "the owner" };
    const i = h.neighbours.findIndex(n => n.visits.includes(src) || n.stored === src);
    if (i >= 0) return { kind: "neighbour", label: `Neighbour ${LETTERS[i]}`, letter: LETTERS[i] };
    if (S.uploads.some(u => u.src === src)) return { kind: "upload", label: "your photo" };
    return { kind: "stranger", label: "a cat from another street" };
  }

  function loadHome(i) {
    S.home = i;
    const h = S.bundle.homes[i];
    S.enrol = h.enrol.map(src => ({ src, on: true }));
    S.memory = h.neighbours.map((n, j) => ({ letter: LETTERS[j], cat: n.cat, stored: n.stored, on: true }));
    const others = S.bundle.homes.filter((_, j) => j !== i).slice(0, 3).map(o => o.owner_visits[0] || o.enrol[0]);
    S.candidates = [
      { key: "owner", label: "Owner", sub: "other visits", srcs: h.owner_visits },
      ...h.neighbours.map((n, j) => ({ key: `n${j}`, label: `Neighbour ${LETTERS[j]}`, sub: "look-alike", srcs: n.visits })),
      { key: "other", label: "Other cats", sub: "random strangers", srcs: others },
    ];
    S.visitor = h.neighbours.flatMap(n => n.visits)[0] || h.owner_visits[0];
    S.log = null;
    renderAll();
  }

  // ---------- rendering ----------
  function thumbButton(src, { pressed, current, tag, dot, label }) {
    const b = document.createElement("button");
    b.className = "thumb-btn";
    b.type = "button";
    if (pressed !== undefined) b.setAttribute("aria-pressed", String(pressed));
    if (current) b.setAttribute("aria-current", "true");
    b.setAttribute("aria-label", label);
    b.innerHTML = `<img src="${S.uploads.find(u => u.src === src)?.dataUrl || "bundle/" + src}" alt="" loading="lazy">` +
      (tag ? `<span class="tag">${tag}</span>` : "") + (dot ? `<span class="mark-dot ${dot}"></span>` : "");
    return b;
  }
  const imgUrl = src => S.uploads.find(u => u.src === src)?.dataUrl || "bundle/" + src;

  function renderHousehold() {
    const h = S.bundle.homes[S.home];
    const on = S.enrol.filter(e => e.on).length;
    $("#enrol-count").textContent = `${on} of ${S.enrol.length} in use`;
    const enrol = $("#enrol"); enrol.replaceChildren();
    S.enrol.forEach((e, i) => {
      const b = thumbButton(e.src, { pressed: e.on, tag: `Photo ${i + 1}`, label: `Enrolment photo ${i + 1}, ${e.on ? "in use" : "left out"}` });
      b.addEventListener("click", () => {
        if (e.on && S.enrol.filter(x => x.on).length === 1) return toast("Keep at least one enrolment photo.");
        e.on = !e.on; renderAll();
      });
      enrol.append(b);
    });
    const mem = $("#memory"); mem.replaceChildren(); mem.classList.toggle("off", !S.memOn);
    S.memory.forEach(n => {
      const li = document.createElement("li");
      li.innerHTML = `<img src="${imgUrl(n.stored)}" alt=""><div><p class="name">Neighbour ${n.letter}</p><p class="sub">Cat ${n.cat} · kept out before</p></div>
        <input type="checkbox" ${n.on ? "checked" : ""} aria-label="Remember Neighbour ${n.letter}" ${S.memOn ? "" : "disabled"}>`;
      $("input", li).addEventListener("change", ev => { n.on = ev.target.checked; renderAll(); });
      mem.append(li);
    });
    $("#mem-switch").checked = S.memOn;
    const mcfg = S.bundle.models[S.model];
    $("#crumb-home").textContent = h.name;
    $("#door-meta").textContent = `Owner: cat ${h.owner} · ${on} enrolment photo${on > 1 ? "s" : ""} · ${S.memory.filter(n => n.on).length} neighbours remembered`;
    const k = String(Math.min(3, Math.max(1, on))), full = S.rule === "full";
    const m = mcfg.margin[k], memUsed = full && S.memOn && m !== null && m !== undefined;
    $("#settings").innerHTML = `<dt>Model</dt><dd>${mcfg.label} <span>· ${mcfg.detail}</span></dd>
      <dt>Threshold</dt><dd>${fmt(full ? mcfg.t_lookalike[k] : mcfg.t_random)} <span>· ${full ? `tuned on look-alike cats, ${k} photo${k > 1 ? "s" : ""}` : "1% of random cats get in"}</span></dd>
      <dt>Margin</dt><dd>${memUsed ? signed(m) + " <span>· over the closest remembered neighbour</span>" : "<span>not used</span>"}</dd>`;
  }

  let lastVerdict = "";
  function renderVisit() {
    const d = decide(S.visitor);
    const truth = truthOf(S.visitor);
    const img = $("#visit-img");
    const url = imgUrl(S.visitor);
    if (img.dataset.src !== url) {
      img.classList.add("swap");
      img.onload = () => requestAnimationFrame(() => img.classList.remove("swap"));
      img.src = url; img.dataset.src = url;
    }
    img.style.filter = S.cam === "door" ? "grayscale(0.6) blur(0.6px) brightness(0.7) contrast(0.9)" : "";
    img.alt = `Visitor: ${truth.label}`;
    $("#visit-cap").textContent = S.cam === "door"
      ? "Door camera: the model scores a small, blurred, dark, compressed copy of this photo."
      : "Photo from the door, scored as is.";

    // check 1
    const p1 = clamp(d.so, 0, 1);
    setPill("#c1-pill", d.pass1 ? "pass" : "fail", d.pass1 ? "Passes" : "Fails");
    $("#c1-dot").style.left = `${p1 * 100}%`;
    $("#c1-tick").style.left = `${clamp(d.t, 0, 1) * 100}%`;
    const f1 = $("#c1-fill"); f1.style.transform = `scaleX(${p1})`; f1.style.background = d.pass1 ? "var(--positive)" : "var(--negative)";
    $("#c1-dot").style.borderColor = d.pass1 ? "var(--positive)" : "var(--negative)";
    $("#c1-text").textContent = `Owner score ${fmt(d.so)} · needs more than ${fmt(d.t)}`;

    // check 2
    const c2 = $("#check2");
    const top = d.sims[0];
    const diff = top ? d.so - top.s : 0;
    c2.classList.toggle("skipped", !d.useMem);
    const pos = 0.5 + clamp(diff, -0.5, 0.5);
    $("#c2-dot").style.left = `${pos * 100}%`;
    $("#c2-tick").style.left = `${(0.5 + clamp(d.m ?? 0, -0.5, 0.5)) * 100}%`;
    const f2 = $("#c2-fill");
    if (diff >= 0) { f2.style.left = "50%"; f2.style.transformOrigin = "left center"; }
    else { f2.style.left = "0"; f2.style.transformOrigin = "right center"; }
    f2.style.transform = `scaleX(${d.useMem ? Math.abs(clamp(diff, -0.5, 0.5)) / 0.5 : 0})`;
    const col2 = d.pass2 === null ? "var(--control)" : d.pass2 ? "var(--positive)" : "var(--negative)";
    f2.style.background = col2; $("#c2-dot").style.borderColor = d.pass2 === null ? "var(--muted)" : col2;
    if (d.useMem) {
      setPill("#c2-pill", d.pass2 ? "pass" : "fail", d.pass2 ? "Passes" : "Fails");
      $("#c2-text").textContent = `Closest remembered: Neighbour ${top.n.letter} at ${fmt(top.s)} · owner minus neighbour ${signed(diff)}, needs more than ${signed(d.m)}`;
    } else {
      setPill("#c2-pill", "skip", "Not used");
      $("#c2-text").textContent = d.why2;
    }

    // verdict
    const v = $("#verdict");
    v.classList.toggle("open", d.open); v.classList.toggle("closed", !d.open);
    $("#verdict-path").setAttribute("d", d.open ? "M6 12.5l4 4 8-9" : "M7 7l10 10M17 7 7 17");
    const title = d.open ? "Door opens" : "Door stays closed";
    if (title !== lastVerdict) { v.classList.remove("bump"); void v.offsetWidth; v.classList.add("bump"); lastVerdict = title; }
    $("#verdict-title").textContent = title;
    const tr = $("#verdict-truth");
    if (truth.kind === "upload") tr.textContent = d.open ? "Your photo matched this home's owner." : "Your photo did not pass this home's door.";
    else {
      const right = truth.kind === "owner" ? d.open : !d.open;
      const what = truth.kind === "owner"
        ? (d.open ? "the right call" : "the owner was locked out by mistake")
        : (d.open ? `a ${truth.kind === "neighbour" ? "look-alike" : "stranger"} got in` : "kept out, as it should be");
      tr.innerHTML = `It was <b>${truth.label}</b>: <span class="${right ? "right" : "wrong"}">${what}</span>.`;
    }
  }

  function renderPicker() {
    const wrap = $("#pick"); wrap.replaceChildren();
    const marks = new Map((S.log || []).map(r => [r.src, r.open ? "open" : "closed"]));
    const groups = [...S.candidates];
    if (S.uploads.length) groups.push({ key: "up", label: "Your photos", sub: "embedded live", srcs: S.uploads.map(u => u.src) });
    groups.forEach(g => {
      if (!g.srcs.length) return;
      const row = document.createElement("div"); row.className = "pick-row";
      row.innerHTML = `<div class="pick-label"><b>${g.label}</b>${g.sub}</div><div class="pick-thumbs"></div>`;
      g.srcs.forEach((src, i) => {
        const b = thumbButton(src, { current: src === S.visitor, dot: marks.get(src), label: `${g.label}, photo ${i + 1}` });
        b.addEventListener("click", () => { S.visitor = src; renderVisit(); renderPicker(); });
        $(".pick-thumbs", row).append(b);
      });
      wrap.append(row);
    });
  }

  function runAll(animate = true) {
    const rows = [];
    S.candidates.forEach(g => g.srcs.forEach(src => {
      const d = decide(src), t = truthOf(src);
      rows.push({ src, truth: t, so: d.so, top: d.sims[0], open: d.open });
    }));
    S.log = rows;
    renderLog(animate); renderPicker();
  }

  function renderLog(animate) {
    const body = $("#log-body");
    if (!S.log) {
      body.innerHTML = `<tr class="empty"><td colspan="6">No visits yet.</td></tr>`;
      $("#log-summary").textContent = "Run every visit to see how this home's door does.";
      return;
    }
    body.replaceChildren();
    S.log.forEach((r, i) => {
      const right = r.truth.kind === "owner" ? r.open : !r.open;
      const tr = document.createElement("tr");
      if (animate) { tr.className = "new"; tr.style.animationDelay = `${Math.min(i * 30, 300)}ms`; }
      tr.innerHTML = `<td><img src="${imgUrl(r.src)}" alt=""></td>
        <td><span class="who ${r.truth.kind}">${cap(r.truth.label)}</span></td>
        <td class="num">${fmt(r.so)}</td>
        <td class="num">${r.top ? `${r.top.n.letter} · ${fmt(r.top.s)}` : "–"}</td>
        <td><span class="dec ${r.open ? "open" : "closed"}">${r.open ? "Opened" : "Stayed closed"}</span></td>
        <td>${right ? "Correct" : `<span class="dec closed">${r.truth.kind === "owner" ? "Owner locked out" : "Let in by mistake"}</span>`}</td>`;
      tr.addEventListener("click", () => { S.visitor = r.src; renderVisit(); renderPicker(); $("#vis-title").scrollIntoView({ behavior: "smooth", block: "start" }); });
      body.append(tr);
    });
    const count = kind => { const g = S.log.filter(r => r.truth.kind === kind); return [g.filter(r => r.open).length, g.length]; };
    const [oi, on] = count("owner"), [ni, nn] = count("neighbour"), [si, sn] = count("stranger");
    $("#log-summary").innerHTML = `Owner let in <b>${oi} of ${on}</b> visits · look-alikes let in <b>${ni} of ${nn}</b> · other cats let in <b>${si} of ${sn}</b>`;
  }

  function renderAll() {
    renderHousehold(); renderVisit(); renderPicker();
    if (S.log) runAll(false);
  }

  // ---------- helpers ----------
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmt = v => { const s = (Math.round(v * 100) / 100).toFixed(2); return s.startsWith("-") ? "−" + s.slice(1) : s; };
  const signed = v => (v >= 0 ? "+" : "−") + fmt(Math.abs(v));
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  function setPill(sel, cls, text) { const p = $(sel); p.className = `pill ${cls}`; p.textContent = text; }
  let toastTimer;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.hidden = false; t.removeAttribute("data-hide");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.setAttribute("data-hide", ""); setTimeout(() => (t.hidden = true), 220); }, 2800);
  }

  function bindSeg(id, key, after) {
    const seg = $(id);
    $$("button", seg).forEach(b => b.addEventListener("click", () => {
      $$("button", seg).forEach(x => x.setAttribute("aria-checked", String(x === b)));
      S[key] = b.dataset.value; after ? after() : renderAll();
    }));
    seg.addEventListener("keydown", ev => {
      if (!["ArrowLeft", "ArrowRight"].includes(ev.key)) return;
      const bs = $$("button", seg), i = bs.findIndex(b => b.getAttribute("aria-checked") === "true");
      const n = bs[(i + (ev.key === "ArrowRight" ? 1 : bs.length - 1)) % bs.length];
      n.click(); n.focus();
    });
  }

  // ---------- routing ----------
  const TITLES = { door: "Front door", how: "How it works", results: "Results" };
  function route() {
    const r = (location.hash || "#door").slice(1);
    const name = TITLES[r] ? r : "door";
    $$(".screen").forEach(s => (s.hidden = s.dataset.screen !== name));
    $$(".nav-item").forEach(a => a.dataset.route === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
    $("#topbar-title").textContent = TITLES[name];
    document.title = `WhoCat · ${TITLES[name]}`;
    $(".sidebar").classList.remove("open");
    $("#sheet").scrollTop = 0;
  }

  // ---------- uploads (live mode only) ----------
  async function shrink(file) {
    const bmp = await createImageBitmap(file);
    const f = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas"); c.width = Math.round(bmp.width * f); c.height = Math.round(bmp.height * f);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.9);
  }
  async function upload(file) {
    if (!file || !file.type.startsWith("image/")) return toast("Choose a photo file (JPEG or PNG).");
    const btn = $("#upload-btn"); btn.setAttribute("aria-busy", "true");
    toast("Looking for a cat face…");
    try {
      const res = await fetch("/api/embed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: await shrink(file) }) });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || "The server could not read that photo.");
      const src = `upload-${Date.now()}`;
      S.uploads.push({ src, dataUrl: out.crop, emb: out.emb });
      S.visitor = src; renderVisit(); renderPicker();
      toast(out.face_found ? "Face found and cropped." : "No cat face found, so the centre of the photo is used.");
    } catch (err) {
      toast(err.message || "Upload failed. Is demo/server.py still running?");
    } finally { btn.removeAttribute("aria-busy"); }
  }

  // ---------- start ----------
  async function start() {
    window.addEventListener("hashchange", route); route();
    $("#menu-btn").addEventListener("click", () => $(".sidebar").classList.toggle("open"));
    try {
      const res = await fetch("bundle/bundle.json");
      if (!res.ok) throw new Error();
      S.bundle = await res.json();
    } catch {
      $("#door-meta").textContent = "No sample homes found. Run demo/export_bundle.py to create bundle/bundle.json, then reload.";
      $$(".controls button, #run-all").forEach(b => (b.disabled = true));
      return;
    }
    try { const h = await fetch("api/health"); S.live = h.ok && (await h.json()).ok; } catch { S.live = false; }
    const chip = $("#mode-chip"); chip.hidden = false;
    chip.textContent = S.live ? "Live: models running on this machine" : "Sample homes · precomputed embeddings";
    $("#upload-btn").hidden = !S.live;
    $("#upload").addEventListener("change", ev => upload(ev.target.files[0]));

    const sel = $("#home-select");
    S.bundle.homes.forEach((h, i) => sel.add(new Option(h.name, i)));
    sel.addEventListener("change", () => loadHome(+sel.value));
    bindSeg("#model-seg", "model"); bindSeg("#rule-seg", "rule"); bindSeg("#cam-seg", "cam");
    $("#mem-switch").addEventListener("change", ev => { S.memOn = ev.target.checked; renderAll(); });
    $("#run-all").addEventListener("click", () => runAll(true));
    loadHome(0);
  }
  start();
})();
