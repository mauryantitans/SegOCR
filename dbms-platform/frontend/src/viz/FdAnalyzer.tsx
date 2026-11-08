import clsx from "clsx";
import { useMemo, useState } from "react";
import { analyze, closureSteps, decomposeBCNF, fmt, minimalCover, parseAttrs, parseFDs, preservesDependencies, synthesize3NF, type FD } from "./fd";

const showFd = (f: FD) => `${fmt(f.lhs)} → ${fmt(f.rhs)}`;
const NF_ORDER = ["1NF", "2NF", "3NF", "BCNF"];

export default function FdAnalyzer({ attributes = ["StudentID", "Course", "Instructor", "Dept", "Grade"], fds = ["StudentID, Course -> Grade, Instructor", "Instructor -> Dept", "Instructor -> Course"] }: { attributes?: string[]; fds?: string[] }) {
  const [attrText, setAttrText] = useState(attributes.join(", "));
  const [fdText, setFdText] = useState(fds.join("\n"));
  const [x, setX] = useState<string[]>([]);
  const attrs = useMemo(() => parseAttrs(attrText.includes(",") ? attrText : attrText.split("").join(","), undefined), [attrText]);
  const F = useMemo(() => parseFDs(fdText.split("\n"), attrs), [fdText, attrs]);
  const unknown = [...new Set(F.flatMap((f) => [...f.lhs, ...f.rhs]))].filter((a) => !attrs.includes(a));
  const res = useMemo(() => {
    if (unknown.length || attrs.length === 0 || attrs.length > 14) return null;
    const a = analyze(attrs, F);
    const bc = decomposeBCNF(attrs, F);
    return { a, cover: minimalCover(F), tnf: synthesize3NF(attrs, F), bc, pres: preservesDependencies(bc.relations, F) };
  }, [attrs, F, unknown.length]);
  const steps = useMemo(() => (x.length ? closureSteps(x, F) : []), [x, F]);

  return (
    <div className="not-prose my-5 rounded-xl border border-indigo-400/20 bg-ink-900/60 p-4 text-sm">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-indigo-300">Functional dependency & normalization analyzer</div>
      <div className="grid gap-3 md:grid-cols-2">
        <label className="block"><span className="text-xs text-slate-400">Attributes of R (comma-separated)</span><input className="input mt-1 font-mono" value={attrText} onChange={(e) => setAttrText(e.target.value)} /></label>
        <label className="block"><span className="text-xs text-slate-400">Functional dependencies (one per line, e.g. A, B -&gt; C)</span><textarea className="input mt-1 h-24 font-mono" value={fdText} onChange={(e) => setFdText(e.target.value)} /></label>
      </div>
      {unknown.length > 0 && <div className="mt-2 text-xs text-rose-300">Unknown attribute(s) in FDs: {unknown.join(", ")}</div>}
      {attrs.length > 14 && <div className="mt-2 text-xs text-amber-300">Use at most 14 attributes (key search is exponential).</div>}
      {res && (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="space-y-3">
            <div>
              <div className="text-xs text-slate-400">Attribute closure — pick attributes</div>
              <div className="mt-1 flex flex-wrap gap-1">{attrs.map((a) => <button key={a} onClick={() => setX((s) => s.includes(a) ? s.filter((y) => y !== a) : [...s, a])} className={clsx("rounded px-2 py-0.5 font-mono text-xs", x.includes(a) ? "bg-indigo-500/30 text-indigo-100" : "bg-white/5 text-slate-400")}>{a}</button>)}</div>
              {x.length > 0 && (
                <ol className="mt-2 space-y-0.5 font-mono text-[12px] text-slate-300">
                  <li>{"{"}{fmt(x)}{"}"}⁺ starts as {"{"}{fmt(x)}{"}"}</li>
                  {steps.map((s, i) => <li key={i}>apply {showFd(s.fd)} → add {s.added.join(", ")} ⇒ {"{"}{s.result.join(", ")}{"}"}</li>)}
                  <li className={clsx(steps.length ? (steps[steps.length - 1].result.length === attrs.length ? "text-emerald-300" : "text-slate-400") : x.length === attrs.length ? "text-emerald-300" : "text-slate-400")}>
                    {(steps.length ? steps[steps.length - 1].result.length : x.length) === attrs.length ? "= all attributes → superkey" : "not all attributes → not a superkey"}</li>
                </ol>
              )}
            </div>
            <div><div className="text-xs text-slate-400">Candidate keys</div><div className="mt-1 font-mono text-emerald-300">{res.a.keys.map((k) => `{${fmt(k)}}`).join("  ")}</div></div>
            <div className="text-xs text-slate-400">Prime: <span className="font-mono text-slate-200">{fmt(res.a.prime) || "—"}</span> · Non-prime: <span className="font-mono text-slate-200">{fmt(res.a.nonPrime) || "—"}</span></div>
            <div>
              <div className="text-xs text-slate-400">Highest normal form</div>
              <div className="mt-1 flex gap-1">{NF_ORDER.map((n) => <span key={n} className={clsx("chip", NF_ORDER.indexOf(n) <= NF_ORDER.indexOf(res.a.nf) ? "bg-emerald-500/20 text-emerald-200" : "bg-white/5 text-slate-500")}>{n}</span>)}</div>
              <ul className="mt-2 space-y-1 text-[12.5px]">{res.a.violations.map((v, i) => <li key={i} className="text-amber-200"><span className="chip mr-1 bg-amber-500/15">{v.nf} violation</span>{v.reason}</li>)}</ul>
            </div>
          </div>
          <div className="space-y-3">
            <div><div className="text-xs text-slate-400">Minimal (canonical) cover</div><div className="mt-1 font-mono text-[12.5px]">{res.cover.map((f, i) => <div key={i}>{showFd(f)}</div>)}</div></div>
            <div><div className="text-xs text-slate-400">3NF synthesis (lossless, dependency-preserving)</div><div className="mt-1 font-mono text-[12.5px]">{res.tnf.map((r, i) => <div key={i}>R{i + 1}({r.join(", ")})</div>)}</div></div>
            <div>
              <div className="text-xs text-slate-400">BCNF decomposition (lossless; may lose dependencies)</div>
              <ul className="mt-1 space-y-0.5 text-[12px] text-slate-400">{res.bc.steps.map((s, i) => <li key={i}>{s}</li>)}</ul>
              <div className="mt-1 font-mono text-[12.5px]">{res.bc.relations.map((r, i) => <div key={i}>R{i + 1}({r.join(", ")})</div>)}</div>
              <div className={clsx("mt-1 text-xs", res.pres.lost.length ? "text-amber-300" : "text-emerald-300")}>
                {res.pres.lost.length ? `Not dependency-preserving — lost: ${res.pres.lost.map(showFd).join("; ")} (enforce with a trigger/unique constraint across tables or accept 3NF)` : "All dependencies preserved."}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
