/** Functional-dependency algorithms: closure, candidate keys, minimal cover, normal forms, 3NF synthesis, BCNF decomposition. */
export type AttrSet = Set<string>;
export interface FD { lhs: string[]; rhs: string[] }

const sorted = (s: Iterable<string>) => [...s].sort();
export const fmt = (s: Iterable<string>) => sorted(s).join(", ");
const subset = (a: Iterable<string>, b: AttrSet) => [...a].every((x) => b.has(x));

export function parseAttrs(text: string, known?: string[]): string[] {
  const t = text.trim();
  if (!t) return [];
  if (t.includes(",") || t.includes(" ")) return t.split(/[\s,]+/).filter(Boolean);
  if (known && known.every((a) => a.length === 1)) return t.split("");
  return [t];
}

export function parseFDs(lines: string[], attrs: string[]): FD[] {
  return lines.filter((l) => l.includes("->")).map((l) => {
    const [l1, r1] = l.split("->");
    return { lhs: parseAttrs(l1, attrs), rhs: parseAttrs(r1, attrs) };
  });
}

export function closure(x: Iterable<string>, fds: FD[]): AttrSet {
  const c = new Set(x);
  let changed = true;
  while (changed) {
    changed = false;
    for (const f of fds) if (subset(f.lhs, c) && !subset(f.rhs, c)) { f.rhs.forEach((a) => c.add(a)); changed = true; }
  }
  return c;
}

/** Closure with the derivation steps (for teaching). */
export function closureSteps(x: string[], fds: FD[]): { fd: FD; added: string[]; result: string[] }[] {
  const c = new Set(x); const steps = [];
  let changed = true;
  while (changed) {
    changed = false;
    for (const f of fds) {
      if (subset(f.lhs, c)) {
        const added = f.rhs.filter((a) => !c.has(a));
        if (added.length) { added.forEach((a) => c.add(a)); steps.push({ fd: f, added, result: sorted(c) }); changed = true; }
      }
    }
  }
  return steps;
}

function* subsetsBySize(items: string[]): Generator<string[]> {
  const n = items.length;
  for (let size = 1; size <= n; size++) {
    const idx = Array.from({ length: size }, (_, i) => i);
    while (true) {
      yield idx.map((i) => items[i]);
      let i = size - 1;
      while (i >= 0 && idx[i] === n - size + i) i--;
      if (i < 0) break;
      idx[i]++;
      for (let j = i + 1; j < size; j++) idx[j] = idx[j - 1] + 1;
    }
  }
}

export function candidateKeys(attrs: string[], fds: FD[]): string[][] {
  const all = new Set(attrs);
  const onRhs = new Set(fds.flatMap((f) => f.rhs.filter((a) => !f.lhs.includes(a))));
  const core = attrs.filter((a) => !onRhs.has(a)); // must be in every key
  if (closure(core, fds).size === all.size && subset(attrs, closure(core, fds))) return [sorted(core)];
  const rest = attrs.filter((a) => !core.includes(a));
  const keys: string[][] = [];
  for (const extra of subsetsBySize(rest)) {
    const cand = [...core, ...extra];
    if (keys.some((k) => k.every((a) => cand.includes(a)))) continue; // superset of a key
    if (subset(attrs, closure(cand, fds))) keys.push(sorted(cand));
  }
  return keys;
}

export const isSuperkey = (x: Iterable<string>, attrs: string[], fds: FD[]) => subset(attrs, closure(x, fds));

export function splitRhs(fds: FD[]): FD[] {
  return fds.flatMap((f) => f.rhs.filter((a) => !f.lhs.includes(a)).map((a) => ({ lhs: [...f.lhs], rhs: [a] })));
}

export function minimalCover(fds: FD[]): FD[] {
  let g = splitRhs(fds);
  // remove extraneous LHS attributes
  g = g.map((f) => {
    let lhs = [...f.lhs];
    for (const a of [...lhs]) {
      if (lhs.length === 1) break;
      const reduced = lhs.filter((x) => x !== a);
      if (closure(reduced, g).has(f.rhs[0])) lhs = reduced;
    }
    return { lhs: sorted(lhs), rhs: f.rhs };
  });
  // dedupe
  const seen = new Set<string>();
  g = g.filter((f) => { const k = f.lhs.join(",") + "->" + f.rhs[0]; if (seen.has(k)) return false; seen.add(k); return true; });
  // remove redundant FDs
  for (let i = 0; i < g.length; i++) {
    const others = g.filter((_, j) => j !== i);
    if (closure(g[i].lhs, others).has(g[i].rhs[0])) { g.splice(i, 1); i--; }
  }
  return g;
}

export type NF = "1NF" | "2NF" | "3NF" | "BCNF";
export interface Analysis { keys: string[][]; prime: string[]; nonPrime: string[]; nf: NF; violations: { nf: NF; fd: FD; reason: string }[] }

export function analyze(attrs: string[], fds: FD[]): Analysis {
  const keys = candidateKeys(attrs, fds);
  const prime = new Set(keys.flat());
  const nonPrime = attrs.filter((a) => !prime.has(a));
  const violations: Analysis["violations"] = [];
  const single = splitRhs(fds);
  // 2NF: partial dependency of a non-prime attribute on part of a candidate key
  for (const k of keys) {
    if (k.length < 2) continue;
    for (const part of subsetsBySize(k)) {
      if (part.length === k.length) break;
      const c = closure(part, fds);
      const partial = nonPrime.filter((a) => c.has(a));
      if (partial.length) violations.push({ nf: "2NF", fd: { lhs: part, rhs: partial }, reason: `${fmt(partial)} depend(s) on part {${fmt(part)}} of key {${fmt(k)}}` });
    }
  }
  for (const f of single) {
    if (isSuperkey(f.lhs, attrs, fds)) continue;
    const a = f.rhs[0];
    if (!prime.has(a)) violations.push({ nf: "3NF", fd: f, reason: `{${fmt(f.lhs)}} is not a superkey and ${a} is non-prime (transitive dependency)` });
    else violations.push({ nf: "BCNF", fd: f, reason: `{${fmt(f.lhs)}} is not a superkey (allowed in 3NF because ${a} is prime)` });
  }
  const nf: NF = violations.some((v) => v.nf === "2NF") ? "1NF" : violations.some((v) => v.nf === "3NF") ? "2NF" : violations.some((v) => v.nf === "BCNF") ? "3NF" : "BCNF";
  return { keys, prime: sorted(prime), nonPrime, nf, violations };
}

/** FDs of `fds` projected onto `rel` (via closures of all subsets; fine for small teaching relations). */
export function project(rel: string[], fds: FD[]): FD[] {
  const out: FD[] = [];
  for (const x of subsetsBySize(rel)) {
    if (x.length === rel.length) break;
    const c = closure(x, fds);
    const rhs = rel.filter((a) => c.has(a) && !x.includes(a));
    if (rhs.length) out.push({ lhs: x, rhs });
  }
  return minimalCover(out);
}

export function synthesize3NF(attrs: string[], fds: FD[]): string[][] {
  const cover = minimalCover(fds);
  const groups = new Map<string, Set<string>>();
  for (const f of cover) {
    const k = f.lhs.join(",");
    if (!groups.has(k)) groups.set(k, new Set(f.lhs));
    groups.get(k)!.add(f.rhs[0]);
  }
  let rels = [...groups.values()].map((s) => sorted(s));
  const keys = candidateKeys(attrs, fds);
  if (!rels.some((r) => keys.some((k) => k.every((a) => r.includes(a))))) rels.push(keys[0]);
  rels = rels.filter((r, i) => !rels.some((o, j) => j !== i && r.every((a) => o.includes(a)) && (o.length > r.length || j < i)));
  return rels;
}

export function decomposeBCNF(attrs: string[], fds: FD[]): { relations: string[][]; steps: string[] } {
  const steps: string[] = [];
  const work = [sorted(attrs)];
  const done: string[][] = [];
  while (work.length) {
    const r = work.pop()!;
    const local = project(r, fds);
    const bad = local.find((f) => !subset(r, closure(f.lhs, local)));
    if (!bad) { done.push(r); continue; }
    const c = closure(bad.lhs, local);
    const r1 = r.filter((a) => c.has(a));
    const r2 = r.filter((a) => !c.has(a) || bad.lhs.includes(a));
    steps.push(`R(${fmt(r)}) violates BCNF on ${fmt(bad.lhs)} → ${fmt(bad.rhs)}; split into (${fmt(r1)}) and (${fmt(r2)})`);
    work.push(r1, r2);
  }
  return { relations: done, steps };
}

export function preservesDependencies(rels: string[][], fds: FD[]): { preserved: FD[]; lost: FD[] } {
  const preserved: FD[] = []; const lost: FD[] = [];
  for (const f of splitRhs(fds)) {
    let z = new Set(f.lhs);
    let changed = true;
    while (changed) {
      changed = false;
      for (const r of rels) {
        const t = closure([...z].filter((a) => r.includes(a)), fds);
        for (const a of r) if (t.has(a) && !z.has(a)) { z.add(a); changed = true; }
      }
    }
    (z.has(f.rhs[0]) ? preserved : lost).push(f);
  }
  return { preserved, lost };
}
