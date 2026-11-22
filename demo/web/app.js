/* WhoCat demo: every decision is computed in the browser from 512-d face embeddings.
   Sample homes come from bundle/bundle.json (demo/export_bundle.py); with demo/server.py running,
   your own photos are embedded by the real models through /api/embed. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const LETTERS = "ABCDEFGH";
  const MODELS = ["ours", "petface"], RULES = ["full", "naive"];
  const RULE_NAME = { full: "Full door rule", naive: "Threshold only" };

  const S = {
    bundle: null, live: false, home: 0, model: "ours", rule: "full", cam: "clean",
    enrol: [], memory: [], memOn: true, visitor: null, candidates: [], uploads: [],
  };
  const embCache = new Map();

  // ---------- maths ----------
  function decode(b64) {
    const bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
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
    let n = 0; for (let i = 0; i < d; i++) n += t[i] * t[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < d; i++) t[i] /= n;
    return t;
  }

  /* The door rule of the paper: admit if s_o > t and (with memory) s_o - max_j s_nj > m. */
  function decide(src, model = S.model, rule = S.rule, cam = S.cam) {
    const cfg = S.bundle.models[model];
    const enrolled = S.enrol.filter(e => e.on).map(e => e.src);
    const k = String(Math.min(3, Math.max(1, enrolled.length)));
    const full = rule === "full";
    const t = full ? cfg.t_lookalike[k] : cfg.t_random;
    const m = full ? cfg.margin[k] : null;
    const x = emb(src, model, cam);
    const so = dot(x, template(enrolled, model));
    const mem = S.memory.filter(n => n.on);
    const sims = mem.map(n => ({ n, s: dot(x, emb(n.stored, model, "clean")) })).sort((a, b) => b.s - a.s);
    const pass1 = so > t;
    const useMem = full && S.memOn && mem.length > 0 && m !== null && m !== undefined;
    let why2 = "";
    if (!full) why2 = "Not part of the threshold-only rule.";
    else if (!S.memOn || mem.length === 0) why2 = "Neighbour memory is off.";
    else if (m === null || m === undefined) why2 = "No margin was tuned for this model.";
    const pass2 = useMem ? so - sims[0].s > m : null;
    const open = pass1 && (pass2 === null || pass2);
    return { so, t, m, k, sims, pass1, pass2, useMem, why2, open };
  }

  // ---------- data ----------
  function truthOf(src) {
    const h = S.bundle.homes[S.home];
    if (h.owner_visits.includes(src) || h.enrol.includes(src)) return { kind: "owner", label: "the owner" };
    const i = h.neighbours.findIndex(n => n.visits.includes(src) || n.stored === src);
    if (i >= 0) return { kind: "neighbour", label: `Neighbour ${LETTERS[i]}` };
    if (S.uploads.some(u => u.src === src)) return { kind: "upload", label: "your photo" };
    return { kind: "stranger", label: "a cat from another street" };
  }
  const homeVisits = () => S.candidates.filter(g => g.key !== "other" && g.key !== "up").flatMap(g => g.srcs);

  function setHome(i) {
    S.home = i;
    const h = S.bundle.homes[i];
    S.enrol = h.enrol.map(src => ({ src, on: true }));
    S.memory = h.neighbours.map((n, j) => ({ letter: LETTERS[j], cat: n.cat, stored: n.stored, on: true }));
    const others = S.bundle.homes.filter((_, j) => j !== i).slice(0, 3).map(o => o.owner_visits[0] || o.enrol[0]);
    S.candidates = [
      { key: "owner", label: "Owner", sub: "another day", srcs: h.owner_visits },
      ...h.neighbours.map((n, j) => ({ key: `n${j}`, label: `Neighbour ${LETTERS[j]}`, sub: "look-alike", srcs: n.visits })),
      { key: "other", label: "Other cats", sub: "random", srcs: others },
    ];
  }
  function loadHome(i) {
    setHome(i);
    S.visitor = defaultVisitor();
    $("#home-select").value = String(i);
    renderAll();
  }

  /* Open on the visit that shows the point: a look-alike the plain threshold lets in and the full rule keeps out. */
  function defaultVisitor() {
    const look = S.candidates.filter(g => g.key.startsWith("n")).flatMap(g => g.srcs);
    const hit = look.find(src => decide(src, "ours", "naive").open && !decide(src, "ours", "full").open);
    return hit || look[0] || S.candidates[0].srcs[0];
  }
  /* The first home shown is the one that tells the story best on clear photos with our model. */
  function storyHome() {
    let best = 0, bestScore = -1e9;
    S.bundle.homes.forEach((_, i) => {
      setHome(i);
      const vis = homeVisits();
      const own = vis.filter(s => truthOf(s).kind === "owner"), look = vis.filter(s => truthOf(s).kind === "neighbour");
      const ownIn = own.filter(s => decide(s, "ours", "full").open).length / Math.max(1, own.length);
      const lookFull = look.filter(s => decide(s, "ours", "full").open).length;
      const lookNaive = look.filter(s => decide(s, "ours", "naive").open).length;
      const score = ownIn * 10 - lookFull * 5 + lookNaive;
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return best;
  }

  // ---------- rendering ----------
  const imgUrl = (src, cam = "clean") => {
    const up = S.uploads.find(u => u.src === src);
    if (up) return cam === "door" && up.doorUrl ? up.doorUrl : up.dataUrl;
    return "bundle/" + (cam === "door" && S.bundle.hasDoorImages ? src.replace(/^img\//, "img_door/") : src);
  };
  function thumbButton(src, { pressed, current, tag, dot, label }) {
    const b = document.createElement("button");
    b.className = "thumb-btn";
    b.type = "button";
    if (pressed !== undefined) b.setAttribute("aria-pressed", String(pressed));
    if (current) b.setAttribute("aria-current", "true");
    b.setAttribute("aria-label", label);
    b.innerHTML = `<img src="${imgUrl(src)}" alt="" loading="lazy" decoding="async">` +
      (tag ? `<span class="tag">${tag}</span>` : "") + (dot ? `<span class="mark-dot ${dot}" aria-hidden="true"></span>` : "");
    return b;
  }

  function renderHousehold() {
    const h = S.bundle.homes[S.home];
    const on = S.enrol.filter(e => e.on).length;
    $("#enrol-count").textContent = `${on} of ${S.enrol.length} in use`;
    const enrol = $("#enrol"); enrol.replaceChildren();
    S.enrol.forEach((e, i) => {
      const b = thumbButton(e.src, { pressed: e.on, tag: `Photo ${i + 1}`, label: `Enrolment photo ${i + 1}, ${e.on ? "in use" : "left out"}` });
      b.addEventListener("click", () => {
        if (e.on && S.enrol.filter(x => x.on).length === 1) return toast("The owner needs at least one enrolment photo.");
        e.on = !e.on; renderAll();
      });
      enrol.append(b);
    });
    const mem = $("#memory"); mem.replaceChildren(); mem.classList.toggle("off", !S.memOn);
    S.memory.forEach(n => {
      const li = document.createElement("li");
      li.innerHTML = `<img src="${imgUrl(n.stored)}" alt=""><div><p class="name">Neighbour ${n.letter}</p><p class="sub">Cat ${n.cat}</p></div>
        <label class="mini-switch"><input type="checkbox" ${n.on ? "checked" : ""} ${S.memOn ? "" : "disabled"}><span class="track"></span><span class="sr">Remember Neighbour ${n.letter}</span></label>`;
      $("input", li).addEventListener("change", ev => { n.on = ev.target.checked; renderAll(); });
      mem.append(li);
    });
    $("#mem-switch").checked = S.memOn;
    $("#crumb-home").textContent = h.name;
    const remembered = S.memory.filter(n => n.on).length;
    $("#door-meta").textContent = `Cat ${h.owner} lives here and ${h.neighbours.length} look-alike neighbours come by. All of them are PetFace validation cats the models never trained on.`;
    const mcfg = S.bundle.models[S.model];
    const k = String(Math.min(3, Math.max(1, on))), full = S.rule === "full";
    const m = mcfg.margin[k], memUsed = full && S.memOn && remembered > 0 && m !== null && m !== undefined;
    $("#settings").innerHTML = `<dt>Model</dt><dd>${mcfg.label}<span>${mcfg.detail}</span></dd>
      <dt>Threshold t</dt><dd>${fmt(full ? mcfg.t_lookalike[k] : mcfg.t_random)}<span>${full ? `tuned on look-alike cats, ${k} photo${k > 1 ? "s" : ""}` : "lets in 1% of random cats"}</span></dd>
      <dt>Margin m</dt><dd>${memUsed ? signed(m) : "–"}<span>${memUsed ? `over the closest of ${remembered} remembered neighbour${remembered > 1 ? "s" : ""}` : "not used"}</span></dd>`;
  }

  let lastVerdict = "";
  function renderDoor() {
    const d = decide(S.visitor);
    const truth = truthOf(S.visitor);

    // owner stack
    const stack = $("#owner-stack"); stack.replaceChildren();
    const enrolled = S.enrol.filter(e => e.on);
    enrolled.forEach((e, i) => {
      const im = document.createElement("img"); im.src = imgUrl(e.src); im.alt = "";
      im.style.setProperty("--i", i); stack.append(im);
    });
    stack.style.setProperty("--n", enrolled.length);
    $("#owner-cap").textContent = `${enrolled.length} photo${enrolled.length > 1 ? "s" : ""} averaged`;

    // visitor
    const img = $("#visit-img"), url = imgUrl(S.visitor, S.cam);
    if (img.dataset.src !== url) {
      img.classList.add("swap");
      const done = () => requestAnimationFrame(() => img.classList.remove("swap"));
      img.onload = done; img.onerror = done;
      img.src = url; img.dataset.src = url;
    }
    img.alt = `Visitor at the door: ${truth.label}`;
    $("#visit-cap").textContent = S.cam !== "door" ? "clear photo" : S.bundle.door_images === "exact" ? "door camera, the copy that was scored" : "door camera, simulated";
    const chip = $("#verdict-chip");
    chip.textContent = d.open ? "Opens" : "Stays closed";
    chip.className = `verdict-chip ${d.open ? "open" : "closed"}`;

    // closest neighbour
    const top = d.sims[0];
    const nImg = $("#neigh-img");
    $("#neigh-empty").hidden = !!top;
    nImg.hidden = !top;
    if (top) { nImg.src = imgUrl(top.n.stored); nImg.alt = `Neighbour ${top.n.letter}`; }
    $("#neigh-name").textContent = top ? `Neighbour ${top.n.letter}` : "Neighbour";

    // links
    const sn = top ? top.s : null;
    $("#s-owner").textContent = fmt(d.so);
    $("#s-neigh").textContent = sn === null ? "–" : fmt(sn);
    $("#l-owner").style.transform = `scaleX(${clamp(d.so, 0, 1)})`;
    $("#l-neigh").style.transform = `scaleX(${sn === null ? 0 : clamp(sn, 0, 1)})`;
    const ownerWins = sn === null || d.so >= sn;
    $(".owner-link").classList.toggle("wins", ownerWins);
    $(".neigh-link").classList.toggle("wins", !ownerWins);

    // check 1
    mark("#c1-mark", d.pass1 ? "pass" : "fail");
    $("#c1-val").textContent = `${fmt(d.so)} ${d.pass1 ? ">" : "≤"} ${fmt(d.t)}`;
    const f1 = $("#c1-fill");
    f1.style.transform = `scaleX(${clamp(d.so, 0, 1)})`;
    f1.dataset.state = d.pass1 ? "pass" : "fail";
    $("#c1-tick").style.left = `${clamp(d.t, 0, 1) * 100}%`;
    $("#c1-text").textContent = S.rule === "full"
      ? `Threshold t is tuned so that 1% of look-alike visits pass, for ${d.k} enrolment photo${d.k > 1 ? "s" : ""}.`
      : "Threshold t lets in 1% of random cats: what a door usually ships with.";

    // check 2
    $("#check2").classList.toggle("skipped", !d.useMem);
    const diff = top ? d.so - top.s : 0;
    const f2 = $("#c2-fill");
    if (diff >= 0) { f2.style.left = "50%"; f2.style.transformOrigin = "left center"; }
    else { f2.style.left = "0"; f2.style.transformOrigin = "right center"; }
    f2.style.transform = `scaleX(${d.useMem ? Math.abs(clamp(diff, -0.5, 0.5)) / 0.5 : 0})`;
    f2.dataset.state = d.pass2 === null ? "skip" : d.pass2 ? "pass" : "fail";
    $("#c2-tick").style.left = `${(0.5 + clamp(d.m ?? 0, -0.5, 0.5)) * 100}%`;
    $("#c2-tick").hidden = !d.useMem;
    if (d.useMem) {
      mark("#c2-mark", d.pass2 ? "pass" : "fail");
      $("#c2-val").textContent = `${signed(diff)} ${d.pass2 ? ">" : "≤"} ${signed(d.m)}`;
      $("#c2-text").textContent = `Owner score ${fmt(d.so)} minus Neighbour ${top.n.letter} ${fmt(top.s)}; margin m must be beaten.`;
    } else {
      mark("#c2-mark", "skip");
      $("#c2-val").textContent = "not used";
      $("#c2-text").textContent = d.why2;
    }

    // verdict
    const v = $("#verdict");
    v.classList.toggle("open", d.open); v.classList.toggle("closed", !d.open);
    $("#verdict-path").setAttribute("d", d.open ? "M6 12.5l4 4 8-9" : "M7 7l10 10M17 7 7 17");
    const title = d.open ? "The door opens" : "The door stays closed";
    if (title !== lastVerdict) { v.classList.remove("bump"); void v.offsetWidth; v.classList.add("bump"); lastVerdict = title; }
    $("#verdict-title").textContent = title;
    const tr = $("#verdict-truth");
    if (truth.kind === "upload") tr.textContent = d.open ? "Your photo matched this home's owner." : "Your photo did not get past this home's door.";
    else {
      const right = truth.kind === "owner" ? d.open : !d.open;
      const what = truth.kind === "owner"
        ? (d.open ? "welcome home" : "the owner is locked out")
        : (d.open ? (truth.kind === "neighbour" ? "a look-alike got in" : "a stranger got in") : "kept out, as it should be");
      tr.innerHTML = `This was <b>${truth.label}</b>: <span class="${right ? "right" : "wrong"}">${what}</span>.`;
    }
  }
  function mark(sel, state) {
    const el = $(sel); el.dataset.state = state;
    el.innerHTML = state === "pass"
      ? '<svg viewBox="0 0 16 16"><path d="M4 8.5l2.5 2.5L12 5.5"/></svg>'
      : state === "fail" ? '<svg viewBox="0 0 16 16"><path d="M5 5l6 6M11 5l-6 6"/></svg>'
      : '<svg viewBox="0 0 16 16"><path d="M5 8h6"/></svg>';
  }

  function renderPicker() {
    const wrap = $("#pick"); wrap.replaceChildren();
    const groups = [...S.candidates];
    if (S.uploads.length) groups.push({ key: "up", label: "Your photos", sub: "embedded live", srcs: S.uploads.map(u => u.src) });
    groups.forEach(g => {
      if (!g.srcs.length) return;
      const row = document.createElement("div"); row.className = "pick-row";
      row.innerHTML = `<div class="pick-label"><b>${g.label}</b> ${g.sub}</div><div class="pick-thumbs"></div>`;
      g.srcs.forEach((src, i) => {
        const open = decide(src).open;
        const b = thumbButton(src, { current: src === S.visitor, dot: open ? "open" : "closed", label: `${g.label}, visit ${i + 1}: ${open ? "door opens" : "door stays closed"}` });
        b.addEventListener("click", () => { S.visitor = src; renderDoor(); renderPicker(); renderLog(); });
        $(".pick-thumbs", row).append(b);
      });
      wrap.append(row);
    });
  }

  function tally(model, rule) {
    const vis = homeVisits();
    const own = vis.filter(s => truthOf(s).kind === "owner").map(s => decide(s, model, rule).open);
    const look = vis.filter(s => truthOf(s).kind === "neighbour").map(s => decide(s, model, rule).open);
    return { own, look };
  }
  function renderBoard() {
    const board = $("#board"); board.replaceChildren();
    const head = document.createElement("div"); head.className = "board-row board-head"; head.setAttribute("role", "row");
    head.innerHTML = `<span role="columnheader"><span class="sr">Model</span></span>${RULES.map(r => `<span role="columnheader">${RULE_NAME[r]}</span>`).join("")}`;
    board.append(head);
    MODELS.forEach(m => {
      const row = document.createElement("div"); row.className = "board-row"; row.setAttribute("role", "row");
      row.innerHTML = `<span class="board-model" role="rowheader">${S.bundle.models[m].label}</span>`;
      RULES.forEach(r => {
        const t = tally(m, r);
        const ownIn = t.own.filter(Boolean).length, lookIn = t.look.filter(Boolean).length;
        const cell = document.createElement("button");
        cell.type = "button"; cell.className = "board-cell"; cell.setAttribute("role", "cell");
        if (m === S.model && r === S.rule) cell.setAttribute("aria-current", "true");
        cell.setAttribute("aria-label", `${S.bundle.models[m].label}, ${RULE_NAME[r]}: owner let in ${ownIn} of ${t.own.length}, look-alikes let in ${lookIn} of ${t.look.length}`);
        cell.innerHTML = `
          <span class="pips" aria-hidden="true">${t.own.map(o => `<i class="pip owner ${o ? "in" : "out"}"></i>`).join("")}<i class="pip-gap"></i>${t.look.map(o => `<i class="pip look ${o ? "in" : "out"}"></i>`).join("")}</span>
          <span class="board-nums"><span class="${ownIn === t.own.length ? "good" : "bad"}">Owner in ${ownIn}/${t.own.length}</span><span class="${lookIn === 0 ? "good" : "bad"}">Look-alikes in ${lookIn}/${t.look.length}</span></span>`;
        cell.addEventListener("click", () => { setSeg("#model-seg", m); setSeg("#rule-seg", r); S.model = m; S.rule = r; renderAll(); });
        row.append(cell);
      });
      board.append(row);
    });
  }

  function renderLog() {
    const body = $("#log-body"); body.replaceChildren();
    homeVisits().forEach(src => {
      const d = decide(src), t = truthOf(src);
      const right = t.kind === "owner" ? d.open : !d.open;
      const tr = document.createElement("tr");
      if (src === S.visitor) tr.setAttribute("aria-current", "true");
      tr.tabIndex = 0;
      tr.innerHTML = `<td><img src="${imgUrl(src, S.cam)}" alt=""></td>
        <td><span class="who ${t.kind}">${cap(t.label)}</span></td>
        <td class="num">${fmt(d.so)}</td>
        <td class="num">${d.sims[0] ? `${fmt(d.sims[0].s)} <span class="muted">${d.sims[0].n.letter}</span>` : "–"}</td>
        <td><span class="dec ${right ? "right" : "wrong"}">${d.open ? "Opened" : "Stayed closed"}</span></td>`;
      const pick = () => { S.visitor = src; renderDoor(); renderPicker(); renderLog(); $("#vis-title").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
      body.append(tr);
    });
    const t = tally(S.model, S.rule);
    $("#tally-sub").textContent = `${S.bundle.models[S.model].label} with the ${RULE_NAME[S.rule].toLowerCase()}, on ${S.cam === "door" ? "door-camera" : "clear"} photos: the owner got in ${t.own.filter(Boolean).length} of ${t.own.length} times and look-alikes ${t.look.filter(Boolean).length} of ${t.look.length} times. Pick a cell to switch.`;
  }

  function renderAll() { renderHousehold(); renderDoor(); renderPicker(); renderBoard(); renderLog(); }

  // ---------- helpers ----------
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmt = v => { const s = (Math.round(v * 100) / 100).toFixed(2); return s.startsWith("-") ? "−" + s.slice(1) : s; };
  const signed = v => (v >= 0 ? "+" : "−") + fmt(Math.abs(v));
  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  let toastTimer;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.hidden = false;
    requestAnimationFrame(() => t.removeAttribute("data-hide"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.setAttribute("data-hide", ""); setTimeout(() => (t.hidden = true), 220); }, 3000);
  }
  function setSeg(id, value) {
    $$("button", $(id)).forEach(x => { const on = x.dataset.value === value; x.setAttribute("aria-checked", String(on)); x.tabIndex = on ? 0 : -1; });
  }
  function bindSeg(id, key) {
    const seg = $(id);
    $$("button", seg).forEach(b => b.addEventListener("click", () => { setSeg(id, b.dataset.value); S[key] = b.dataset.value; renderAll(); }));
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
    $(".sidebar").classList.remove("open"); $("#menu-btn").setAttribute("aria-expanded", "false");
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
      const res = await fetch("api/embed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: await shrink(file) }) });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || "The server could not read that photo.");
      const src = `upload-${Date.now()}`;
      S.uploads.push({ src, dataUrl: out.crop, doorUrl: out.door_crop, emb: out.emb });
      S.visitor = src; renderDoor(); renderPicker();
      toast(out.face_found ? "Cat face found and cropped." : "No cat face found, so the centre of the photo is used.");
    } catch (err) {
      toast(err.message || "Upload failed. Is demo/server.py still running?");
    } finally { btn.removeAttribute("aria-busy"); $("#upload").value = ""; }
  }

  // ---------- start ----------
  async function start() {
    window.addEventListener("hashchange", route); route();
    $("#menu-btn").addEventListener("click", () => {
      const open = $(".sidebar").classList.toggle("open");
      $("#menu-btn").setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("keydown", ev => { if (ev.key === "Escape") $(".sidebar").classList.remove("open"); });
    try {
      const res = await fetch("bundle/bundle.json");
      if (!res.ok) throw new Error();
      S.bundle = await res.json();
    } catch {
      $("#door-meta").textContent = "No sample homes found. Run demo/export_bundle.py to create bundle/bundle.json, then reload.";
      $$(".controls button").forEach(b => (b.disabled = true));
      return;
    }
    S.bundle.hasDoorImages = S.bundle.door_images !== false;  // "exact" when exported with the scored copies
    try { const h = await fetch("api/health"); S.live = h.ok && (await h.json()).ok === true; } catch { S.live = false; }
    const chip = $("#mode-chip"); chip.hidden = false;
    chip.textContent = S.live ? "Live: models running on this machine" : `${S.bundle.homes.length} sample homes · scores from the trained models`;
    $("#upload-btn").hidden = !S.live;
    $("#upload").addEventListener("change", ev => upload(ev.target.files[0]));

    const sel = $("#home-select");
    S.bundle.homes.forEach((h, i) => sel.add(new Option(h.name, i)));
    sel.addEventListener("change", () => loadHome(+sel.value));
    bindSeg("#model-seg", "model"); bindSeg("#rule-seg", "rule"); bindSeg("#cam-seg", "cam");
    $("#mem-switch").addEventListener("change", ev => { S.memOn = ev.target.checked; renderAll(); });
    loadHome(storyHome());
    document.body.classList.add("ready");
  }
  start();
})();
