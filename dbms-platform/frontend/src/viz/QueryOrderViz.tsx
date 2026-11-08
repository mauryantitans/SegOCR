import clsx from "clsx";
import { useState } from "react";

const CLAUSES = [
  { c: "FROM / JOIN", logical: 1, d: "Build the working set: read the source tables and combine them with joins (ON conditions apply here)." },
  { c: "WHERE", logical: 2, d: "Filter individual rows. Aggregates and SELECT aliases do not exist yet." },
  { c: "GROUP BY", logical: 3, d: "Partition the remaining rows into groups." },
  { c: "HAVING", logical: 4, d: "Filter whole groups using aggregates." },
  { c: "SELECT", logical: 5, d: "Evaluate the output expressions and aliases (window functions are computed here, after HAVING)." },
  { c: "DISTINCT", logical: 6, d: "Remove duplicate output rows." },
  { c: "ORDER BY", logical: 7, d: "Sort the result. SELECT aliases are visible here." },
  { c: "LIMIT / OFFSET", logical: 8, d: "Return only the requested slice." },
];
const WRITTEN = ["SELECT", "DISTINCT", "FROM / JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT / OFFSET"];

export default function QueryOrderViz() {
  const [view, setView] = useState<"written" | "logical">("logical");
  const [sel, setSel] = useState<string | null>("WHERE");
  const order = view === "written" ? WRITTEN : [...CLAUSES].sort((a, b) => a.logical - b.logical).map((x) => x.c);
  return (
    <div className="not-prose my-5 rounded-xl border border-emerald-400/20 bg-ink-900/60 p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300">Query clause order</span>
        {(["written", "logical"] as const).map((v) => <button key={v} onClick={() => setView(v)} className={clsx("rounded-md px-2.5 py-1 text-xs", view === v ? "bg-emerald-500/20 text-emerald-200" : "text-slate-400 hover:bg-white/5")}>{v === "written" ? "As you write it" : "As it is logically evaluated"}</button>)}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {order.map((c, i) => (
          <div key={c} className="flex items-center gap-1.5" style={{ transition: "all .4s" }}>
            <button onClick={() => setSel(c)} className={clsx("rounded-lg border px-3 py-1.5 font-mono text-xs transition", sel === c ? "border-emerald-300 bg-emerald-500/20 text-emerald-100" : "border-white/10 text-slate-300 hover:border-white/25")}>
              <span className="mr-1.5 text-slate-500">{i + 1}</span>{c}</button>
            {i < order.length - 1 && <span className="text-slate-600">→</span>}
          </div>
        ))}
      </div>
      {sel && <p className="mt-3 text-sm text-slate-300"><span className="font-mono text-emerald-300">{sel}</span> — {CLAUSES.find((x) => x.c === sel)?.d}</p>}
      <p className="mt-2 text-[11px] text-slate-500">This is the logical (semantic) order defined by SQL. The optimizer may physically reorder work — push filters into joins, use an index to avoid sorting, stop early for LIMIT — as long as the result is the same. EXPLAIN FORMAT=TREE shows the physical plan.</p>
    </div>
  );
}
