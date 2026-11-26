// LipSpell browser engine (pure ES module, zero dependencies)

export class DistinctiveFeatures {
  constructor(data) {
    this.featureNames = data.features;
    this.phones = data.phones;
    this.vectors = data.vectors;
    this.phoneToId = new Map();
    this.phones.forEach((p, i) => this.phoneToId.set(p, i));

    const n = this.phones.length;
    const f = this.featureNames.length;
    const weights = this.featureNames.map((name) => data.salience[name] ?? 1.0);
    const maxTotal = weights.reduce((a, b) => a + b, 0);

    // Norms for cosine distance
    const norms = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      const v = this.vectors[i];
      for (let k = 0; k < f; k++) s += v[k] * v[k];
      norms[i] = Math.sqrt(s);
    }

    this.subMatrix = Array.from({ length: n }, () => new Float32Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dot = 0;
        const vi = this.vectors[i];
        const vj = this.vectors[j];
        for (let k = 0; k < f; k++) dot += vi[k] * vj[k];
        const denom = norms[i] * norms[j];
        const cos = denom > 0 ? dot / denom : 0;
        const d = Math.max(0, Math.min(1, 0.5 * (1 - cos)));
        this.subMatrix[i][j] = d;
        this.subMatrix[j][i] = d;
      }
    }

    this.indelCosts = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      const v = this.vectors[i];
      for (let k = 0; k < f; k++) s += Math.abs(v[k]) * weights[k];
      const raw = (s / maxTotal) * 1.4;
      this.indelCosts[i] = Math.max(0.35, Math.min(0.75, raw));
    }

    this.sortedPhones = [...this.phones].sort((a, b) => b.length - a.length);
    this.coarseClass = this._buildCoarseClasses();
  }

  _buildCoarseClasses() {
    const idx = {};
    this.featureNames.forEach((name, i) => (idx[name] = i));
    const classes = {};
    for (let pid = 0; pid < this.phones.length; pid++) {
      const p = this.phones[pid];
      const v = this.vectors[pid];
      if (v[idx.syll] > 0) classes[p] = "V";
      else if (v[idx.nas] > 0) classes[p] = "N";
      else if (v[idx.son] > 0 && v[idx.cont] > 0) classes[p] = "L";
      else if (v[idx.strid] > 0) classes[p] = "S";
      else if (v[idx.lab] > 0) classes[p] = "P";
      else if (v[idx.cor] > 0) classes[p] = "T";
      else classes[p] = "K";
    }
    return classes;
  }

  tokenize(ipa) {
    const ids = [];
    let i = 0;
    while (i < ipa.length) {
      let matched = false;
      for (const p of this.sortedPhones) {
        if (ipa.startsWith(p, i)) {
          ids.push(this.phoneToId.get(p));
          i += p.length;
          matched = true;
          break;
        }
      }
      if (!matched) i += 1;
    }
    return ids;
  }

  skeleton(phoneIds) {
    let out = "";
    let prev = "";
    for (const pid of phoneIds) {
      const cls = this.coarseClass[this.phones[pid]];
      if (cls !== prev) {
        out += cls;
        prev = cls;
      }
    }
    return out;
  }
}

const ROW_OFFSETS = [0.0, 0.5, 0.75, 1.25];

export class KeyboardLayout {
  constructor(rows, saturation = 3.0) {
    this.saturation = saturation;
    this.coords = new Map();
    for (let r = 0; r < rows.length; r++) {
      const off = ROW_OFFSETS[r] ?? r * 0.5;
      const row = rows[r];
      for (let c = 0; c < row.length; c++) {
        this.coords.set(row[c], [r, c + off]);
      }
    }
    this.chars = [...this.coords.keys()];
    this.charToId = new Map();
    this.chars.forEach((ch, i) => this.charToId.set(ch, i));
    const n = this.chars.length;
    this.matrix = Array.from({ length: n }, () => new Float32Array(n));
    for (let i = 0; i < n; i++) {
      const [x1, y1] = this.coords.get(this.chars[i]);
      for (let j = i + 1; j < n; j++) {
        const [x2, y2] = this.coords.get(this.chars[j]);
        const dist = Math.min(Math.hypot(x1 - x2, y1 - y2), saturation) / saturation;
        this.matrix[i][j] = dist;
        this.matrix[j][i] = dist;
      }
    }
  }

  static qwerty() {
    return new KeyboardLayout(["1234567890-=", "qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]);
  }

  static abnt2() {
    return new KeyboardLayout(["'1234567890-=", "qwertyuiop´[", "asdfghjklç~]", "\\zxcvbnm,.;/"]);
  }

  distance(c1, c2) {
    const a = c1.toLowerCase();
    const b = c2.toLowerCase();
    if (a === b) return 0.0;
    const b1 = a.normalize("NFD");
    const b2 = b.normalize("NFD");
    if (b1 && b2 && b1[0] === b2[0] && (b1.length > 1 || b2.length > 1)) {
      return 0.15;
    }
    const i = this.charToId.get(a);
    const j = this.charToId.get(b);
    if (i === undefined || j === undefined) return 1.0;
    return this.matrix[i][j];
  }
}

const PT_MULTI = [
  ["ch", "ʃ"], ["lh", "ʎ"], ["nh", "ɲ"], ["rr", "h"], ["ss", "s"],
  ["qu", "k"], ["gu", "ɡ"], ["ão", "aʊ"], ["õe", "oɪ"], ["ãe", "aɪ"],
  ["ou", "o"], ["ei", "eɪ"], ["ai", "aɪ"], ["oi", "ɔɪ"], ["au", "aʊ"],
];
const PT_VOWELS = new Set("aeiouáàâãéêíóôõú".split(""));
const PT_SINGLE = {
  a: "a", á: "a", à: "a", â: "ɐ", ã: "ɐ",
  b: "b", c: "k", ç: "s", d: "d",
  e: "e", é: "ɛ", è: "e", ê: "e",
  f: "f", g: "ɡ", h: "",
  i: "i", í: "i", ì: "i", j: "ʒ", k: "k", l: "l", m: "m", n: "n",
  o: "o", ó: "ɔ", ò: "o", ô: "o", õ: "o",
  p: "p", q: "k", r: "ɹ", s: "s", t: "t",
  u: "u", ú: "u", ù: "u", v: "v", w: "w", x: "ʃ", y: "i", z: "z",
};

export class PortugueseG2P {
  constructor() { this.cache = new Map(); }
  transcribe(word) {
    const w0 = word.toLowerCase();
    if (this.cache.has(w0)) return this.cache.get(w0);
    let w = "";
    for (const ch of w0) if (ch in PT_SINGLE) w += ch;
    if (!w) return "";
    const out = [];
    const n = w.length;
    let i = 0;
    while (i < n) {
      if ((w.startsWith("sc", i) || w.startsWith("xc", i)) && i + 2 < n && "eéêií".includes(w[i + 2])) {
        out.push("s");
        i += 2;
        continue;
      }
      let matched = false;
      for (const [pat, repl] of PT_MULTI) {
        if (w.startsWith(pat, i)) {
          out.push(repl);
          i += pat.length;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      const ch = w[i];
      if (ch === "c" && i + 1 < n && "eéêií".includes(w[i + 1])) out.push("s");
      else if (ch === "g" && i + 1 < n && "eéêií".includes(w[i + 1])) out.push("ʒ");
      else if (ch === "s" && i > 0 && i < n - 1 && PT_VOWELS.has(w[i - 1]) && PT_VOWELS.has(w[i + 1])) out.push("z");
      else if (ch === "z" && i === n - 1) out.push("s");
      else if (ch === "r" && i === 0) out.push("h");
      else out.push(PT_SINGLE[ch] ?? "");
      i += 1;
    }
    const res = out.join("");
    this.cache.set(w0, res);
    return res;
  }
}

const EN_MULTI = [
  ["tion", "ʃən"], ["sion", "ʒən"], ["ough", "oʊ"], ["augh", "ɔ"],
  ["eigh", "eɪ"], ["igh", "aɪ"], ["tch", "tʃ"], ["dge", "dʒ"],
  ["sch", "sk"], ["qu", "kw"], ["ph", "f"], ["sh", "ʃ"],
  ["ch", "tʃ"], ["th", "θ"], ["wh", "w"], ["ng", "ŋ"],
  ["ck", "k"], ["dg", "dʒ"], ["gh", ""], ["kn", "n"],
  ["gn", "n"], ["wr", "ɹ"], ["ps", "s"], ["ee", "i"],
  ["ea", "i"], ["oo", "u"], ["ou", "aʊ"], ["ow", "oʊ"],
  ["oi", "ɔɪ"], ["oy", "ɔɪ"], ["ai", "eɪ"], ["ay", "eɪ"],
  ["ei", "eɪ"], ["ey", "eɪ"], ["au", "ɔ"], ["aw", "ɔ"],
  ["ar", "ɑɹ"], ["er", "ɚ"], ["ir", "ɝ"], ["or", "ɔɹ"], ["ur", "ɝ"],
];
const EN_SINGLE = {
  a: "æ", b: "b", c: "k", d: "d", e: "ɛ", f: "f", g: "ɡ", h: "h",
  i: "ɪ", j: "dʒ", k: "k", l: "l", m: "m", n: "n", o: "ɑ", p: "p",
  q: "k", r: "ɹ", s: "s", t: "t", u: "ʌ", v: "v", w: "w", x: "ks",
  y: "j", z: "z",
};
const EN_LONG_E = { a: "eɪ", e: "i", i: "aɪ", o: "oʊ", u: "u" };

export class EnglishG2P {
  constructor() { this.cache = new Map(); }
  transcribe(word) {
    const w0 = word.toLowerCase();
    if (this.cache.has(w0)) return this.cache.get(w0);
    const w = w0.replace(/[^a-z]/g, "");
    if (!w) return "";
    const out = [];
    const n = w.length;
    let i = 0;
    while (i < n) {
      if (i === n - 1 && w[i] === "e" && n > 3 && !"aeiou".includes(w[i - 1])) {
        i += 1;
        continue;
      }
      let matched = false;
      for (const [pat, repl] of EN_MULTI) {
        if (w.startsWith(pat, i)) {
          out.push(repl);
          i += pat.length;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      const ch = w[i];
      if (ch in EN_LONG_E && i + 2 === n - 1 && !"aeiou".includes(w[i + 1]) && w[i + 2] === "e") {
        out.push(EN_LONG_E[ch]);
        i += 1;
        continue;
      }
      if (ch === "c" && i + 1 < n && "eiy".includes(w[i + 1])) out.push("s");
      else if (ch === "g" && i + 1 < n && "eiy".includes(w[i + 1])) out.push("dʒ");
      else out.push(EN_SINGLE[ch] ?? "");
      if (i + 1 < n && w[i + 1] === ch && ch !== "e" && ch !== "o") i += 2;
      else i += 1;
    }
    const res = out.join("");
    this.cache.set(w0, res);
    return res;
  }
}

export function orthographicDL(s1, s2) {
  const n1 = s1.length, n2 = s2.length;
  if (n1 === 0) return n2;
  if (n2 === 0) return n1;
  let prev2 = new Int32Array(n2 + 1);
  let prev1 = new Int32Array(n2 + 1);
  let curr = new Int32Array(n2 + 1);
  for (let j = 0; j <= n2; j++) prev1[j] = j;
  for (let i = 1; i <= n1; i++) {
    curr[0] = i;
    const c1 = s1[i - 1];
    for (let j = 1; j <= n2; j++) {
      const c2 = s2[j - 1];
      const cost = c1 === c2 ? 0 : 1;
      let best = Math.min(prev1[j] + 1, curr[j - 1] + 1, prev1[j - 1] + cost);
      if (i > 1 && j > 1 && c1 === s2[j - 2] && s1[i - 2] === c2) {
        best = Math.min(best, prev2[j - 2] + 1);
      }
      curr[j] = best;
    }
    [prev2, prev1, curr] = [prev1, curr, prev2];
  }
  return prev1[n2];
}

export function keyboardDL(s1, s2, kb) {
  const n1 = s1.length, n2 = s2.length;
  if (n1 === 0) return n2;
  if (n2 === 0) return n1;
  let prev2 = new Float64Array(n2 + 1);
  let prev1 = new Float64Array(n2 + 1);
  let curr = new Float64Array(n2 + 1);
  for (let j = 0; j <= n2; j++) prev1[j] = j;
  for (let i = 1; i <= n1; i++) {
    curr[0] = i;
    const c1 = s1[i - 1];
    let delCost = 0.7;
    if (i > 1) delCost = Math.min(delCost, 0.4 + 0.6 * kb.distance(c1, s1[i - 2]));
    if (i < n1) delCost = Math.min(delCost, 0.4 + 0.6 * kb.distance(c1, s1[i]));
    for (let j = 1; j <= n2; j++) {
      const c2 = s2[j - 1];
      const subCost = c1 === c2 ? 0.0 : kb.distance(c1, c2);
      let insCost = 0.7;
      if (j > 1) insCost = Math.min(insCost, 0.4 + 0.6 * kb.distance(c2, s2[j - 2]));
      if (j < n2) insCost = Math.min(insCost, 0.4 + 0.6 * kb.distance(c2, s2[j]));
      let best = Math.min(prev1[j] + delCost, curr[j - 1] + insCost, prev1[j - 1] + subCost);
      if (i > 1 && j > 1 && c1 === s2[j - 2] && s1[i - 2] === c2) {
        best = Math.min(best, prev2[j - 2] + 0.45);
      }
      curr[j] = best;
    }
    [prev2, prev1, curr] = [prev1, curr, prev2];
  }
  return prev1[n2];
}

export function phoneticDL(p1, p2, subMatrix, indelCosts) {
  const n1 = p1.length, n2 = p2.length;
  if (n1 === 0) {
    let s = 0;
    for (let j = 0; j < n2; j++) s += indelCosts[p2[j]];
    return s;
  }
  if (n2 === 0) {
    let s = 0;
    for (let i = 0; i < n1; i++) s += indelCosts[p1[i]];
    return s;
  }
  let prev2 = new Float64Array(n2 + 1);
  let prev1 = new Float64Array(n2 + 1);
  let curr = new Float64Array(n2 + 1);
  for (let j = 1; j <= n2; j++) prev1[j] = prev1[j - 1] + indelCosts[p2[j - 1]];
  for (let i = 1; i <= n1; i++) {
    const id1 = p1[i - 1];
    const dCost = indelCosts[id1];
    curr[0] = prev1[0] + dCost;
    const subRow = subMatrix[id1];
    for (let j = 1; j <= n2; j++) {
      const id2 = p2[j - 1];
      const iCost = indelCosts[id2];
      const sCost = id1 === id2 ? 0.0 : subRow[id2];
      let best = Math.min(prev1[j] + dCost, curr[j - 1] + iCost, prev1[j - 1] + sCost);
      if (i > 1 && j > 1 && id1 === p2[j - 2] && p1[i - 2] === id2) {
        best = Math.min(best, prev2[j - 2] + 0.45);
      }
      curr[j] = best;
    }
    [prev2, prev1, curr] = [prev1, curr, prev2];
  }
  return prev1[n2];
}

function generateDeletes(word, maxDist) {
  const deletes = new Set([word]);
  let queue = [word];
  for (let step = 0; step < maxDist; step++) {
    const nextQ = [];
    for (const item of queue) {
      if (item.length <= 1) continue;
      for (let i = 0; i < item.length; i++) {
        const d = item.slice(0, i) + item.slice(i + 1);
        if (!deletes.has(d)) {
          deletes.add(d);
          nextQ.push(d);
        }
      }
    }
    queue = nextQ;
  }
  return deletes;
}

export class LipSpellEngine {
  constructor(entries, features, g2p, keyboard) {
    this.features = features;
    this.g2p = g2p;
    this.keyboard = keyboard;
    this.counts = new Map();
    this.vocab = new Set();
    this.deleteIndex = new Map();
    this.skelIndex = new Map();
    this.wordIpa = new Map();
    this.wordPhones = new Map();
    this.wordSkel = new Map();

    // entries must be sorted descending by count so buckets are naturally frequency-ordered
    let totalCount = 0;
    for (const [word, count] of entries) {
      const w = word.toLowerCase();
      if (!w || this.vocab.has(w)) continue;
      this.vocab.add(w);
      this.counts.set(w, count);
      totalCount += count;

      for (const d of generateDeletes(w, 2)) {
        let bucket = this.deleteIndex.get(d);
        if (!bucket) {
          bucket = [];
          this.deleteIndex.set(d, bucket);
        }
        bucket.push(w);
      }

      const ipa = this.g2p.transcribe(w);
      const pids = this.features.tokenize(ipa);
      const skel = this.features.skeleton(pids);
      this.wordIpa.set(w, ipa);
      this.wordPhones.set(w, pids);
      this.wordSkel.set(w, skel);

      if (skel) {
        for (const sKey of generateDeletes(skel, 1)) {
          let bucket = this.skelIndex.get(sKey);
          if (!bucket) {
            bucket = [];
            this.skelIndex.set(sKey, bucket);
          }
          if (bucket.length < 300) bucket.push(w);
        }
      }
    }
    this.logDenom = Math.log(totalCount + 0.5 * this.vocab.size);
  }

  logProb(word) {
    return Math.log((this.counts.get(word) ?? 0) + 0.5) - this.logDenom;
  }

  queryCandidates(word, mode) {
    const cands = new Set();
    if (mode === "norvig" || mode === "keyboard" || mode === "unified") {
      for (const d of generateDeletes(word, 2)) {
        const hits = this.deleteIndex.get(d);
        if (hits) for (const h of hits) cands.add(h);
      }
    }
    if (mode === "phonetic" || mode === "unified") {
      const pids = this.features.tokenize(this.g2p.transcribe(word));
      const skel = this.features.skeleton(pids);
      if (skel) {
        for (const sKey of generateDeletes(skel, 1)) {
          const hits = this.skelIndex.get(sKey);
          if (hits) for (const h of hits) cands.add(h);
        }
      }
    }
    return cands;
  }

  neighbors(word, { mode = "unified", weights = [1.0, 4.0, 1.5, 4.0], topK = 20, excludeSelf = true } = {}) {
    const wLow = word.trim().toLowerCase();
    if (!wLow) return { query: null, results: [] };

    const qIpa = this.g2p.transcribe(wLow);
    const qPhones = this.features.tokenize(qIpa);
    const qSkel = this.features.skeleton(qPhones);
    const inVocab = this.vocab.has(wLow);

    const cands = this.queryCandidates(wLow, mode);
    if (excludeSelf) cands.delete(wLow);

    const [wFreq, wOrth, wKb, wPhon] = weights;
    const scored = [];

    for (const c of cands) {
      const lp = this.logProb(c);
      const dOrth = orthographicDL(wLow, c);
      if (mode === "norvig" && dOrth > 2) continue;
      const dKb = keyboardDL(wLow, c, this.keyboard);
      const cPhones = this.wordPhones.get(c) ?? [];
      const dPhon = phoneticDL(qPhones, cPhones, this.features.subMatrix, this.features.indelCosts);

      let score;
      if (mode === "norvig") {
        score = -100.0 * dOrth + lp;
      } else if (mode === "phonetic") {
        score = wFreq * lp - wPhon * dPhon;
      } else if (mode === "keyboard") {
        score = wFreq * lp - wKb * dKb;
      } else {
        score = wFreq * lp - wOrth * dOrth - wKb * dKb - wPhon * dPhon;
      }

      scored.push({
        word: c,
        count: this.counts.get(c) ?? 0,
        ipa: this.wordIpa.get(c) ?? "",
        skeleton: this.wordSkel.get(c) ?? "",
        dOrth,
        dKb,
        dPhon,
        logP: lp,
        score,
      });
    }

    scored.sort((a, b) => (b.score !== a.score ? b.score - a.score : a.word.localeCompare(b.word)));
    return {
      query: {
        word: wLow,
        inVocab,
        count: this.counts.get(wLow) ?? 0,
        ipa: qIpa,
        skeleton: qSkel,
        candidateCount: cands.size,
      },
      results: scored.slice(0, topK),
    };
  }
}
