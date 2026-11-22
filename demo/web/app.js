/* WhoCat demo: every decision is computed in the browser from 512-d face embeddings.
   Sample homes come from bundle/bundle.json (demo/export_bundle.py). With demo/server.py running,
   photos you take or upload are embedded by the real models through /api/embed. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const LETTERS = "ABCDEFGHIJKLMNOP";
  const RULE_NAME = { full: "Full door rule", naive: "Threshold only" };
  const DOOR_NAME = {
    "petface|naive": ["Normal door", "PetFace model, one threshold"],
    "ours|full": ["WhoCat door", "Our model, look-alike threshold and neighbour memory"],
    "ours|naive": ["Our model, plain threshold", "Our model, one threshold"],
    "petface|full": ["PetFace with our rule", "PetFace model, look-alike threshold and neighbour memory"],
  };
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  const S = {
    bundle: null, live: false, mode: "sample", cam: "clean", memOn: true,
    doors: [{ model: "petface", rule: "naive" }, { model: "ours", rule: "full" }],
    uploads: [],
    sample: { home: 0, enrol: [], memory: [], groups: [], visitor: null },
    mine: { enrol: [], memory: [], tests: [], strangers: [], visitor: null },
  };
  const ctx = () => (S.mode === "mine" ? S.mine : S.sample);
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
  const tplCache = new Map();
  function template(srcs, model) {
    const key = model + "|" + srcs.join(",");
    if (tplCache.has(key)) return tplCache.get(key);
    const d = S.bundle.dim, t = new Float32Array(d);
    srcs.forEach(src => { const e = emb(src, model, "clean"); for (let i = 0; i < d; i++) t[i] += e[i]; });
    let n = 0; for (let i = 0; i < d; i++) n += t[i] * t[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < d; i++) t[i] /= n;
    tplCache.set(key, t);
    return t;
  }

  /* The door rule of the paper: admit if s_o > t and (with memory) s_o - max_j s_nj > m.
     The plain door has only the first check, with t set so that 1% of random cats get in. */
  function decide(src, door, cam = S.cam, c = ctx()) {
    const cfg = S.bundle.models[door.model];
    const enrolled = c.enrol.filter(e => e.on).map(e => e.src);
    if (!enrolled.length || !src) return null;
    const k = String(Math.min(3, enrolled.length));
    const full = door.rule === "full";
    const t = full ? cfg.t_lookalike[k] : cfg.t_random;
    const m = full ? cfg.margin[k] : null;
    const x = emb(src, door.model, cam);
    const so = dot(x, template(enrolled, door.model));
    const mem = c.memory.filter(n => n.on && n.stored !== src);
    const sims = mem.map(n => ({ n, s: dot(x, emb(n.stored, door.model, "clean")) })).sort((a, b) => b.s - a.s);
    const pass1 = so > t;
    const useMem = full && S.memOn && sims.length > 0 && m !== null && m !== undefined;
    let why2 = "";
    if (!full) why2 = "This door has no second check.";
    else if (!S.memOn) why2 = "Neighbour memory is switched off.";
    else if (!sims.length) why2 = "No neighbour remembered yet.";
    const pass2 = useMem ? so - sims[0].s > m : null;
    const open = pass1 && (pass2 === null || pass2);
    return { so, t, m, k, sims, pass1, pass2, useMem, why2, open };
  }

  // ---------- who is who ----------
  function truthOf(src) {
    if (S.mode === "mine") {
      if (S.mine.strangers.includes(src)) return { kind: "stranger", label: "a cat from another home" };
      return { kind: "upload", label: "your photo" };
    }
    const h = S.bundle.homes[S.sample.home];
    if (h.owner_visits.includes(src) || h.enrol.includes(src)) return { kind: "owner", label: "the owner" };
    const i = h.neighbours.findIndex(n => n.visits.includes(src) || n.stored === src);
    if (i >= 0) return { kind: "neighbour", label: `Neighbour ${LETTERS[i]}` };
    return { kind: "stranger", label: "a cat from another street" };
  }
  const rightCall = (kind, open) => (kind === "owner" ? open : kind === "upload" ? null : !open);
  function groups() {
    if (S.mode === "sample") return S.sample.groups;
    return [
      { key: "tests", label: "Your visitors", sub: S.mine.tests.length ? "photos you took" : "none yet", srcs: S.mine.tests },
      { key: "strangers", label: "Cats from other homes", sub: "should stay out", srcs: S.mine.strangers },
    ];
  }
  const allVisits = () => groups().flatMap(g => g.srcs);

  // ---------- homes ----------
  function setHome(i) {
    const h = S.bundle.homes[i], c = S.sample;
    c.home = i;
    c.enrol = h.enrol.map(src => ({ src, on: true }));
    c.memory = h.neighbours.map((n, j) => ({ label: `Neighbour ${LETTERS[j]}`, sub: `cat ${n.cat}`, stored: n.stored, on: true }));
    const others = S.bundle.homes.filter((_, j) => j !== i).slice(0, 3).map(o => o.owner_visits[0] || o.enrol[0]);
    c.groups = [
      { key: "owner", label: "Owner", sub: "on another day", srcs: h.owner_visits },
      ...h.neighbours.map((n, j) => ({ key: `n${j}`, label: `Neighbour ${LETTERS[j]}`, sub: "look-alike", srcs: n.visits })),
      { key: "other", label: "Other cats", sub: "random", srcs: others },
    ];
  }
  function loadHome(i) {
    setHome(i);
    S.sample.visitor = storyVisitor();
    $("#home-select").value = String(i);
    renderAll();
  }
  /* Open on the visit that makes the point: a look-alike the normal door lets in and the WhoCat door keeps out. */
  function storyVisitor() {
    const [a, b] = S.doors;
    const look = S.sample.groups.filter(g => g.key.startsWith("n")).flatMap(g => g.srcs);
    const sd = (src, dr) => decide(src, dr, S.cam, S.sample);
    const hit = look.find(src => sd(src, a).open && !sd(src, b).open);
    return hit || look[0] || S.sample.groups[0].srcs[0];
  }
  function storyHome() {
    const [a, b] = S.doors;
    let best = 0, bestScore = -1e9;
    S.bundle.homes.forEach((_, i) => {
      setHome(i);
      const h = S.bundle.homes[i];
      const own = h.owner_visits, look = h.neighbours.flatMap(n => n.visits);
      const sd = (src, dr) => decide(src, dr, "clean", S.sample).open;
      const ownIn = own.filter(s => sd(s, b)).length / Math.max(1, own.length);
      const score = ownIn * 10 - look.filter(s => sd(s, b)).length * 5 + look.filter(s => sd(s, a)).length;
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return best;
  }

  // ---------- rendering ----------
  const imgUrl = (src, cam = "clean") => {
    const up = S.uploads.find(u => u.src === src);
    if (up) return cam === "door" ? up.doorUrl : up.dataUrl;
    return "bundle/" + (cam === "door" && S.bundle.hasDoorImages ? src.replace(/^img\//, "img_door/") : src);
  };
  const doorName = d => DOOR_NAME[`${d.model}|${d.rule}`];

  function renderMode() {
    const mine = S.mode === "mine";
    $("#mine").hidden = !mine;
    $("#mine-live").hidden = !S.live;
    $("#mine-off").hidden = S.live;
    $("#visit-btns").hidden = !(mine && S.live);
    $(".scope").hidden = mine;
    const h = S.bundle.homes[S.sample.home];
    $("#crumb").textContent = mine ? "Your own cat / Front door" : `${h.name} / Front door`;
    $("#visits-sub").textContent = mine
      ? "Photograph your cat on another day, another cat, or a person holding a photo. Cats from the sample homes are here too, to see strangers kept out."
      : `Cat ${h.owner} lives here and ${h.neighbours.length} look-alike neighbour${h.neighbours.length > 1 ? "s" : ""} come by. All are PetFace validation cats the models never trained on.`;
  }

  function renderMine() {
    if (S.mode !== "mine" || !S.live) return;
    const wrap = $("#mine-enrol"); wrap.replaceChildren();
    for (let i = 0; i < 3; i++) {
      const e = S.mine.enrol[i];
      const slot = document.createElement("div"); slot.className = "slot";
      if (e) {
        slot.innerHTML = `<img src="${imgUrl(e.src)}" alt="Enrolment photo ${i + 1}"><button class="icon-btn on-img" type="button" aria-label="Remove enrolment photo ${i + 1}"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15"/></svg></button>`;
        $("button", slot).addEventListener("click", () => { S.mine.enrol.splice(i, 1); tplCache.clear(); renderAll(); });
      } else {
        slot.classList.add("empty");
        slot.innerHTML = `<span>${i === 0 ? "First photo" : "Optional"}</span>`;
      }
      wrap.append(slot);
    }
    $("#mine-title").textContent = S.mine.enrol.length ? `Your cat, ${S.mine.enrol.length} photo${S.mine.enrol.length > 1 ? "s" : ""} enrolled` : "Enrol your cat";
    $$('[data-capture="enrol"], [data-upload="enrol"]').forEach(b => {
      const full = S.mine.enrol.length >= 3;
      (b.tagName === "INPUT" ? b.closest("label") : b).classList.toggle("disabled", full);
      b.disabled = full;
    });
  }

  function renderFaces() {
    const c = ctx();
    const stack = $("#owner-stack"); stack.replaceChildren();
    const enrolled = c.enrol.filter(e => e.on);
    enrolled.forEach((e, i) => {
      const im = document.createElement("img"); im.src = imgUrl(e.src); im.alt = ""; im.style.setProperty("--i", i); stack.append(im);
    });
    stack.style.setProperty("--n", Math.max(1, enrolled.length));
    stack.classList.toggle("empty", !enrolled.length);
    $("#owner-cap").textContent = enrolled.length ? `${enrolled.length} photo${enrolled.length > 1 ? "s" : ""} averaged` : "not enrolled yet";

    const img = $("#visit-img"), empty = $("#visit-empty");
    const v = c.visitor;
    img.hidden = !v; empty.hidden = !!v;
    if (!v) {
      empty.textContent = S.mode === "mine" && !S.live ? "Needs the live server" : !enrolled.length ? "Enrol your cat first" : "Photograph a visitor below";
      $("#visit-cap").textContent = "nobody yet";
    } else {
      const url = imgUrl(v, S.cam);
      if (img.dataset.src !== url) {
        if (!reduced()) img.classList.add("swap");
        const done = () => requestAnimationFrame(() => img.classList.remove("swap"));
        img.onload = done; img.onerror = done;
        img.src = url; img.dataset.src = url;
      }
      img.alt = `Visitor at the door: ${truthOf(v).label}`;
      $("#visit-cap").textContent = S.cam === "door" ? "as the door camera sees it" : "clear photo";
    }

    const d = v && decide(v, S.doors[1]);
    const top = d && d.sims[0];
    const nImg = $("#neigh-img");
    $("#neigh-empty").hidden = !!top; nImg.hidden = !top;
    if (top) { nImg.src = imgUrl(top.n.stored); nImg.alt = top.n.label; }
    $("#neigh-name").textContent = top ? top.n.label : "Closest neighbour";
  }

  const MARK = {
    pass: '<svg viewBox="0 0 16 16"><path d="M4 8.5l2.5 2.5L12 5.5"/></svg>',
    fail: '<svg viewBox="0 0 16 16"><path d="M5 5l6 6M11 5l-6 6"/></svg>',
    skip: '<svg viewBox="0 0 16 16"><path d="M5 8h6"/></svg>',
    enter: '<svg viewBox="0 0 16 16"><path d="M2.5 8h7M7 5l3 3-3 3M12.5 3v10"/></svg>',
    lock: '<svg viewBox="0 0 16 16"><rect x="3.5" y="7" width="9" height="6.5" rx="1.5"/><path d="M5.5 7V5.2a2.5 2.5 0 0 1 5 0V7"/></svg>',
  };
  function doorCard(door, i) {
    const v = ctx().visitor, d = v && decide(v, door);
    const [name, sub] = doorName(door);
    const el = document.createElement("article");
    el.className = "door-card" + (i === 1 ? " ours" : "");
    if (!d) {
      el.innerHTML = `<header><h3>${name}</h3><p>${sub}</p></header><div class="dverdict idle"><span class="dicon">${MARK.skip}</span><span>Waiting for a visitor</span></div>`;
      return el;
    }
    const truth = truthOf(v), right = rightCall(truth.kind, d.open);
    const top = d.sims[0], diff = top ? d.so - top.s : 0;
    const c1 = clamp(d.so, 0, 1), t1 = clamp(d.t, 0, 1);
    const half = 0.5, c2 = d.useMem ? clamp(diff, -half, half) : 0, t2 = d.useMem ? clamp(d.m, -half, half) : 0;
    const outcome = right === null ? (d.open ? "It thinks this is your cat." : "It does not think this is your cat.")
      : truth.kind === "owner" ? (d.open ? "Right: the owner is home." : "Wrong: the owner is locked out.")
      : d.open ? (truth.kind === "neighbour" ? "Wrong: a look-alike got in." : "Wrong: a stranger got in.") : "Right: kept out.";
    el.innerHTML = `
      <header><h3>${name}</h3><p>${sub}</p></header>
      <div class="dverdict ${d.open ? "open" : "closed"}"><span class="dicon">${d.open ? MARK.enter : MARK.lock}</span><span>${d.open ? "Opens" : "Stays closed"}</span></div>
      <p class="doutcome ${right === null ? "" : right ? "right" : "wrong"}">${right === null ? "" : `<span class="omark">${right ? MARK.pass : MARK.fail}</span>`}${outcome}</p>
      <div class="dcheck">
        <div class="dcheck-head"><span class="cmark" data-state="${d.pass1 ? "pass" : "fail"}">${d.pass1 ? MARK.pass : MARK.fail}</span><span class="cname">Looks like the owner</span><span class="cval">${fmt(d.so)} ${d.pass1 ? "&gt;" : "≤"} ${fmt(d.t)}</span></div>
        <div class="bar"><span class="bar-fill" data-state="${d.pass1 ? "pass" : "fail"}" style="transform:scaleX(${c1})"></span><span class="bar-tick" style="left:${t1 * 100}%"></span></div>
        <p class="cnote">${door.rule === "full" ? `Similarity to the owner template must beat t, tuned so 1% of look-alike visits pass (${d.k} photo${d.k > 1 ? "s" : ""}).` : "Similarity to the owner must beat t, set so 1% of random cats pass."}</p>
      </div>
      <div class="dcheck ${d.useMem ? "" : "skipped"}">
        <div class="dcheck-head"><span class="cmark" data-state="${d.pass2 === null ? "skip" : d.pass2 ? "pass" : "fail"}">${d.pass2 === null ? MARK.skip : d.pass2 ? MARK.pass : MARK.fail}</span><span class="cname">More like the owner than any neighbour</span><span class="cval">${d.useMem ? `${signed(diff)} ${d.pass2 ? "&gt;" : "≤"} ${signed(d.m)}` : "not used"}</span></div>
        <div class="bar centred"><span class="bar-mid"></span><span class="bar-fill" data-state="${d.pass2 === null ? "skip" : d.pass2 ? "pass" : "fail"}" style="${c2 >= 0 ? "left:50%;transform-origin:left center" : "left:0;transform-origin:right center"};transform:scaleX(${Math.abs(c2) / half})"></span>${d.useMem ? `<span class="bar-tick" style="left:${(0.5 + t2) * 100}%"></span>` : ""}</div>
        <p class="cnote">${d.useMem ? `Owner ${fmt(d.so)} minus ${top.n.label} ${fmt(top.s)} must beat the margin m.` : d.why2}</p>
      </div>`;
    return el;
  }
  let lastKey = "";
  function renderDoors() {
    const wrap = $("#doors"), cards = S.doors.map(doorCard);
    const v = ctx().visitor, key = v + S.cam + S.doors.map(d => d.model + d.rule).join();
    wrap.replaceChildren(...cards);
    if (key !== lastKey && !reduced()) cards.forEach(c => { const dv = $(".dverdict", c); dv && dv.classList.add("bump"); });
    lastKey = key;

    const truth = $("#truth"); truth.replaceChildren();
    if (!v) return;
    const t = truthOf(v);
    if (S.mode === "mine" && t.kind === "upload") {
      const remembered = S.mine.memory.some(n => n.stored === v);
      truth.innerHTML = remembered ? `<p>The door remembers this cat as a neighbour. Visitors that look more like it than like your cat are kept out.</p>` : `<p>Not your cat? Let the door remember it as a neighbour, as it does with every look-alike it keeps out. The WhoCat door then checks each visitor against it.</p>`;
      const b = document.createElement("button"); b.type = "button"; b.className = "btn";
      b.textContent = remembered ? "Forget this neighbour" : "Remember as a neighbour";
      b.addEventListener("click", () => {
        if (remembered) S.mine.memory = S.mine.memory.filter(n => n.stored !== v);
        else S.mine.memory.push({ label: `Neighbour ${LETTERS[S.mine.memory.length]}`, sub: "from your photos", stored: v, on: true });
        S.mine.memory.forEach((n, j) => (n.label = `Neighbour ${LETTERS[j]}`));
        renderAll();
      });
      truth.append(b);
    } else {
      truth.innerHTML = `<p>This visitor was <b>${t.label}</b>${t.kind === "neighbour" ? ", a look-alike of the owner" : ""}.</p>`;
    }
  }

  function renderVisits() {
    const wrap = $("#pick"); wrap.replaceChildren();
    const c = ctx();
    groups().forEach(g => {
      if (!g.srcs.length && g.key !== "tests") return;
      const row = document.createElement("div"); row.className = "pick-row";
      row.innerHTML = `<div class="pick-label"><b>${g.label}</b> ${g.sub}</div><div class="pick-thumbs"></div>`;
      g.srcs.forEach((src, i) => {
        const ds = S.doors.map(dr => decide(src, dr));
        const t = truthOf(src);
        const b = document.createElement("button"); b.type = "button"; b.className = "thumb-btn";
        if (src === c.visitor) b.setAttribute("aria-current", "true");
        const word = d => (d ? (d.open ? "opens" : "stays closed") : "no decision");
        b.setAttribute("aria-label", `${g.label}, visit ${i + 1}: ${doorName(S.doors[0])[0]} ${word(ds[0])}, ${doorName(S.doors[1])[0]} ${word(ds[1])}`);
        const dotCls = d => { if (!d) return "none"; const r = rightCall(t.kind, d.open); return r === null ? (d.open ? "open" : "closed") : r ? "right" : "wrong"; };
        const remembered = c.memory.some(n => n.stored === src && n.on);
        b.innerHTML = `<img src="${imgUrl(src, S.cam)}" alt="" loading="lazy" decoding="async">${remembered ? '<span class="tag">remembered</span>' : ""}<span class="dots" aria-hidden="true"><i class="${dotCls(ds[0])}"></i><i class="${dotCls(ds[1])}"></i></span>`;
        b.addEventListener("click", () => { c.visitor = src; renderStage(); renderVisits(); });
        $(".pick-thumbs", row).append(b);
      });
      if (!g.srcs.length) $(".pick-thumbs", row).innerHTML = `<p class="hint">${c.enrol.length ? "Use “Photo a visitor” above." : "Enrol your cat first, then photograph a visitor."}</p>`;
      wrap.append(row);
    });

    const tallies = $("#tallies"); tallies.replaceChildren();
    $("#legend").innerHTML = S.mode === "sample"
      ? `Under each photo, left dot: <b>${doorName(S.doors[0])[0]}</b>; right dot: <b>${doorName(S.doors[1])[0]}</b>. <span class="lg right"></span>right call <span class="lg wrong"></span>wrong call`
      : `Under each photo, left dot: <b>${doorName(S.doors[0])[0]}</b>; right dot: <b>${doorName(S.doors[1])[0]}</b>. <span class="lg open"></span>opens <span class="lg closed"></span>stays closed <span class="lg right"></span>stranger kept out <span class="lg wrong"></span>stranger let in`;
    if (S.mode !== "sample") return;
    const vis = c.groups.filter(g => g.key !== "other").flatMap(g => g.srcs);
    const own = vis.filter(s => truthOf(s).kind === "owner"), look = vis.filter(s => truthOf(s).kind === "neighbour");
    S.doors.forEach((door, i) => {
      const oi = own.filter(s => decide(s, door).open).length, li = look.filter(s => decide(s, door).open).length;
      const el = document.createElement("div"); el.className = "tally" + (i === 1 ? " ours" : "");
      el.innerHTML = `<p class="tname">${doorName(door)[0]}</p>
        <p class="tstat"><span class="big ${oi === own.length ? "" : "bad"}">${oi}/${own.length}</span> owner visits let in</p>
        <p class="tstat"><span class="big ${li === 0 ? "" : "bad"}">${li}/${look.length}</span> look-alike visits let in</p>`;
      tallies.append(el);
    });
  }

  function renderAdjust() {
    const c = ctx();
    const on = c.enrol.filter(e => e.on).length;
    $("#enrol-count").textContent = c.enrol.length ? `${on} of ${c.enrol.length} in use` : "";
    const enrol = $("#enrol"); enrol.replaceChildren();
    if (!c.enrol.length) enrol.innerHTML = '<p class="hint">No enrolment photos yet.</p>';
    c.enrol.forEach((e, i) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "thumb-btn";
      b.setAttribute("aria-pressed", String(e.on));
      b.setAttribute("aria-label", `Enrolment photo ${i + 1}, ${e.on ? "in use" : "left out"}`);
      b.innerHTML = `<img src="${imgUrl(e.src)}" alt=""><span class="tag">${e.on ? "in use" : "left out"}</span>`;
      b.addEventListener("click", () => {
        if (e.on && c.enrol.filter(x => x.on).length === 1) return toast("The owner needs at least one enrolment photo.");
        e.on = !e.on; renderAll();
      });
      enrol.append(b);
    });
    const mem = $("#memory"); mem.replaceChildren(); mem.classList.toggle("off", !S.memOn);
    if (!c.memory.length) mem.innerHTML = `<li class="hint">${S.mode === "mine" ? "Pick a visitor and choose “Remember as a neighbour”." : "This home has no neighbours."}</li>`;
    c.memory.forEach(n => {
      const li = document.createElement("li");
      li.innerHTML = `<img src="${imgUrl(n.stored)}" alt=""><div><p class="name">${n.label}</p><p class="sub">${n.sub}</p></div>
        <label class="mini-switch"><input type="checkbox" ${n.on ? "checked" : ""} ${S.memOn ? "" : "disabled"}><span class="track"></span><span class="sr">Remember ${n.label}</span></label>`;
      $("input", li).addEventListener("change", ev => { n.on = ev.target.checked; renderAll(); });
      mem.append(li);
    });
    $("#mem-switch").checked = S.memOn;

    const cfg = $("#door-cfg"); cfg.replaceChildren();
    S.doors.forEach((door, i) => {
      const k = String(Math.min(3, Math.max(1, on))), mc = S.bundle.models[door.model];
      const t = door.rule === "full" ? mc.t_lookalike[k] : mc.t_random, m = door.rule === "full" ? mc.margin[k] : null;
      const row = document.createElement("div"); row.className = "cfg-row";
      row.innerHTML = `<p class="cfg-name">${i === 0 ? "Left" : "Right"} door <span class="hint">t ${fmt(t)}${m !== null ? `, m ${signed(m)}` : ""}</span></p>
        <div class="seg small" role="radiogroup" aria-label="${i === 0 ? "Left" : "Right"} door face model">
          <button role="radio" data-value="petface">PetFace model</button><button role="radio" data-value="ours">Our model</button></div>
        <div class="seg small" role="radiogroup" aria-label="${i === 0 ? "Left" : "Right"} door rule">
          <button role="radio" data-value="naive">Threshold only</button><button role="radio" data-value="full">Full door rule</button></div>`;
      const [segM, segR] = $$(".seg", row);
      wireSeg(segM, door.model, val => { door.model = val; renderAll(); });
      wireSeg(segR, door.rule, val => { door.rule = val; renderAll(); });
      cfg.append(row);
    });
  }

  function renderStage() { renderFaces(); renderDoors(); }
  function renderAll() { tplCache.clear(); renderMode(); renderMine(); renderStage(); renderVisits(); renderAdjust(); }

  // ---------- helpers ----------
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmt = v => { const s = (Math.round(v * 100) / 100).toFixed(2); return s.startsWith("-") ? "−" + s.slice(1) : s; };
  const signed = v => (v >= 0 ? "+" : "−") + fmt(Math.abs(v));
  let toastTimer;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.hidden = false;
    requestAnimationFrame(() => t.removeAttribute("data-hide"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.setAttribute("data-hide", ""); setTimeout(() => (t.hidden = true), 220); }, 3200);
  }
  function wireSeg(seg, value, onPick) {
    const bs = $$("button", seg);
    const set = v => bs.forEach(x => { const on = x.dataset.value === v; x.setAttribute("aria-checked", String(on)); x.tabIndex = on ? 0 : -1; });
    set(value);
    bs.forEach(b => b.addEventListener("click", () => { set(b.dataset.value); onPick(b.dataset.value); }));
    seg.addEventListener("keydown", ev => {
      if (!["ArrowLeft", "ArrowRight"].includes(ev.key)) return;
      const i = bs.findIndex(b => b.getAttribute("aria-checked") === "true");
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

  // ---------- photos: embed through the live server ----------
  async function shrinkFile(file) {
    const bmp = await createImageBitmap(file);
    return shrinkSource(bmp, bmp.width, bmp.height);
  }
  function shrinkSource(srcEl, w, h) {
    const f = Math.min(1, 1280 / Math.max(w, h));
    const c = document.createElement("canvas"); c.width = Math.round(w * f); c.height = Math.round(h * f);
    c.getContext("2d").drawImage(srcEl, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.92);
  }
  async function embedPhoto(dataUrl, purpose) {
    const res = await fetch("api/embed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: dataUrl }) });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(out.error || "The server could not read that photo.");
    const src = `upload-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    S.uploads.push({ src, dataUrl: out.crop, doorUrl: out.door_crop, emb: out.emb });
    if (purpose === "enrol") {
      S.mine.enrol.push({ src, on: true });
    } else {
      S.mine.tests.unshift(src);
      S.mine.visitor = src;
    }
    renderAll();
    if (purpose === "visit") $("#stage-title").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
    toast(out.face_found ? "Cat face found and cropped." : "No cat face found, so the centre of the photo is used. Try closer, eyes level.");
  }
  async function handleFile(file, purpose, btn) {
    if (!file) return;
    if (!file.type.startsWith("image/")) return toast("Choose a photo file (JPEG or PNG).");
    btn && btn.setAttribute("aria-busy", "true");
    toast("Looking for a cat face…");
    try { await embedPhoto(await shrinkFile(file), purpose); }
    catch (err) { toast(err.message || "Upload failed. Is demo/server.py still running?"); }
    finally { btn && btn.removeAttribute("aria-busy"); }
  }

  // ---------- camera ----------
  const cam = { stream: null, facing: "environment", purpose: "visit", shot: null, devices: 0 };
  async function startStream() {
    stopStream();
    const msg = $("#cam-msg"); msg.hidden = true;
    try {
      cam.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: cam.facing }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
      const v = $("#cam-video"); v.srcObject = cam.stream; await v.play().catch(() => {});
      const devs = await navigator.mediaDevices.enumerateDevices();
      cam.devices = devs.filter(d => d.kind === "videoinput").length;
      $("#cam-flip").hidden = cam.devices < 2;
      $("#cam-snap").disabled = false;
    } catch (err) {
      msg.hidden = false;
      msg.textContent = err && err.name === "NotAllowedError"
        ? "The browser blocked the camera. Allow it in the address bar, or use Upload instead."
        : "No camera found. Use Upload instead.";
      $("#cam-snap").disabled = true;
      $("#cam-flip").hidden = true;
    }
  }
  function stopStream() { if (cam.stream) cam.stream.getTracks().forEach(t => t.stop()); cam.stream = null; }
  function showShot(on) {
    $("#cam-shot").hidden = !on; $("#cam-video").hidden = on;
    $("#cam-snap").hidden = on; $("#cam-use").hidden = !on; $("#cam-retake").hidden = !on;
    $(".cam-guide").hidden = on;
  }
  function openCamera(purpose) {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return toast("The camera only works when the page is opened as http://localhost. Use Upload instead.");
    }
    cam.purpose = purpose;
    $("#cam-title").textContent = purpose === "enrol" ? "Photo of your cat" : "Who's at the door?";
    $("#cam-hint").textContent = purpose === "enrol"
      ? "Fill the square with your cat's face, eyes level, in good light."
      : "Photograph the visitor's face: your cat on another day, or a different cat.";
    showShot(false);
    $("#cam-use").removeAttribute("aria-busy");
    const dlg = $("#cam"); dlg.showModal();
    startStream();
  }
  function closeCamera() {
    stopStream();
    const dlg = $("#cam");
    if (!dlg.open) return;
    if (reduced()) return dlg.close();
    dlg.classList.add("closing");
    setTimeout(() => { dlg.classList.remove("closing"); dlg.close(); }, 150);
  }
  function snap() {
    const v = $("#cam-video");
    if (!v.videoWidth) return;
    cam.shot = shrinkSource(v, v.videoWidth, v.videoHeight);
    $("#cam-shot").src = cam.shot;
    showShot(true);
  }
  async function useShot() {
    const btn = $("#cam-use"); btn.setAttribute("aria-busy", "true"); btn.textContent = "Finding the face…";
    try { await embedPhoto(cam.shot, cam.purpose); closeCamera(); }
    catch (err) { toast(err.message || "The server did not answer. Is demo/server.py running?"); }
    finally { btn.removeAttribute("aria-busy"); btn.textContent = "Use this photo"; }
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
      $("#doors").innerHTML = '<p class="hint">No sample homes found. Put the exported bundle in demo/web/bundle (demo/export_bundle.py writes it), then reload.</p>';
      $$("#mode-seg button, #cam-seg button").forEach(b => (b.disabled = true));
      return;
    }
    S.bundle.hasDoorImages = S.bundle.door_images !== false;
    try { const h = await fetch("api/health", { cache: "no-store" }); S.live = h.ok && (await h.json()).ok === true; } catch { S.live = false; }
    const chip = $("#mode-chip"); chip.hidden = false;
    chip.textContent = S.live ? "Live: the models are running on this computer" : `${S.bundle.homes.length} sample homes, scored by the trained models`;
    chip.classList.toggle("live", S.live);

    // strangers for "your own cat": owners of the sample homes
    S.mine.strangers = S.bundle.homes.slice(0, 6).map(h => h.owner_visits[0] || h.enrol[0]);

    const sel = $("#home-select");
    S.bundle.homes.forEach((h, i) => sel.add(new Option(h.name, i)));
    sel.addEventListener("change", () => loadHome(+sel.value));
    if (/[?&]mode=mine/.test(location.search)) S.mode = "mine";
    wireSeg($("#mode-seg"), S.mode, v => { S.mode = v; renderAll(); });
    wireSeg($("#cam-seg"), S.cam, v => { S.cam = v; renderStage(); renderVisits(); });
    $("#mem-switch").addEventListener("change", ev => { S.memOn = ev.target.checked; renderAll(); });

    $$("[data-capture]").forEach(b => b.addEventListener("click", () => openCamera(b.dataset.capture)));
    $$("[data-upload]").forEach(inp => inp.addEventListener("change", ev => { handleFile(ev.target.files[0], inp.dataset.upload, inp.closest("label")); inp.value = ""; }));
    $("#cam-close").addEventListener("click", closeCamera);
    $("#cam").addEventListener("cancel", ev => { ev.preventDefault(); closeCamera(); });
    $("#cam").addEventListener("click", ev => { if (ev.target === ev.currentTarget) closeCamera(); });
    $("#cam-snap").addEventListener("click", snap);
    $("#cam-retake").addEventListener("click", () => showShot(false));
    $("#cam-use").addEventListener("click", useShot);
    $("#cam-flip").addEventListener("click", () => { cam.facing = cam.facing === "user" ? "environment" : "user"; startStream(); });

    loadHome(storyHome());
    document.body.classList.add("ready");
  }
  start();
})();
