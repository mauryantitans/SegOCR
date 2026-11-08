import clsx from "clsx";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type Cell = string | number | null;
interface Table { name: string; columns: string[]; rows: Cell[][] }
type JoinType = "inner" | "left" | "right" | "full" | "cross";
const DESC: Record<JoinType, string> = {
  inner: "Only pairs whose join keys are equal. Unmatched rows on either side disappear.",
  left: "Every left row appears; when nothing matches, the right columns are NULL-extended.",
  right: "Every right row appears; when nothing matches, the left columns are NULL-extended.",
  full: "Left join plus the unmatched right rows (MySQL has no FULL OUTER JOIN — emulate with LEFT JOIN … UNION … RIGHT JOIN).",
  cross: "Every left row paired with every right row (Cartesian product): |L| × |R| rows.",
};

const SAMPLE = {
  left: { name: "customers", columns: ["id", "name"], rows: [[1, "Ana"], [2, "Bo"], [3, "Chen"], [4, "Dee"]] },
  right: { name: "orders", columns: ["order_id", "customer_id"], rows: [[101, 1], [102, 1], [103, 3], [104, 9]] },
  leftKey: "id", rightKey: "customer_id",
};

export default function JoinViz(props: { left?: Table; right?: Table; leftKey?: string; rightKey?: string; types?: JoinType[] }) {
  const left = props.left ?? SAMPLE.left, right = props.right ?? SAMPLE.right;
  const lk = left.columns.indexOf(props.leftKey ?? SAMPLE.leftKey), rk = right.columns.indexOf(props.rightKey ?? SAMPLE.rightKey);
  const types = props.types ?? ["inner", "left", "right", "full", "cross"];
  const [type, setType] = useState<JoinType>(types[0]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Build the full sequence of (leftRow, rightRow|null) events, nested-loop style.
  const events = useMemo(() => {
    const ev: { l: number | null; r: number | null; match: boolean; emit: boolean }[] = [];
    const matchedRight = new Set<number>();
    if (type === "right") {
      right.rows.forEach((rr, ri) => {
        let any = false;
        left.rows.forEach((lr, li) => { const m = lr[lk] !== null && lr[lk] === rr[rk]; any ||= m; ev.push({ l: li, r: ri, match: m, emit: m }); });
        if (!any) ev.push({ l: null, r: ri, match: false, emit: true });
      });
      return ev;
    }
    left.rows.forEach((lr, li) => {
      let any = false;
      right.rows.forEach((rr, ri) => {
        const m = type === "cross" || (lr[lk] !== null && lr[lk] === rr[rk]);
        if (m) { any = true; matchedRight.add(ri); }
        ev.push({ l: li, r: ri, match: m, emit: m });
      });
      if (!any && (type === "left" || type === "full")) ev.push({ l: li, r: null, match: false, emit: true });
    });
    if (type === "full") right.rows.forEach((_, ri) => { if (!matchedRight.has(ri)) ev.push({ l: null, r: ri, match: false, emit: true }); });
    return ev;
  }, [type, left, right, lk, rk]);

  useEffect(() => { setStep(0); setPlaying(false); }, [type]);
  useEffect(() => {
    if (!playing) return;
    if (step >= events.length) { setPlaying(false); return; }
    const t = setTimeout(() => setStep((s) => s + 1), 450);
    return () => clearTimeout(t);
  }, [playing, step, events.length]);

  const cur = events[step - 1];
  const result = events.slice(0, step).filter((e) => e.emit).map((e) => [
    ...(e.l === null ? left.columns.map(() => null) : left.rows[e.l]), ...(e.r === null ? right.columns.map(() => null) : right.rows[e.r]),
  ]);
  const T = ({ t, active, side }: { t: Table; active: number | null | undefined; side: "l" | "r" }) => (
    <table className="font-mono text-[12px]">
      <caption className="mb-1 text-left text-xs text-slate-400">{t.name}</caption>
      <thead><tr>{t.columns.map((c, i) => <th key={c} className={clsx("border-b border-white/10 px-2 py-1 text-left", i === (side === "l" ? lk : rk) && type !== "cross" && "text-sky-300")}>{c}</th>)}</tr></thead>
      <tbody>{t.rows.map((r, i) => (
        <tr key={i} className={clsx(active === i && (cur?.match ? "bg-emerald-500/20" : "bg-white/10"))}>
          {r.map((v, j) => <td key={j} className="px-2 py-0.5">{v === null ? <i className="text-slate-500">NULL</i> : v}</td>)}</tr>))}</tbody>
    </table>
  );

  return (
    <div className="not-prose my-5 rounded-xl border border-emerald-400/20 bg-ink-900/60 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <span className="mr-2 text-xs font-semibold uppercase tracking-wider text-emerald-300">Join visualizer</span>
        {types.map((t) => <button key={t} onClick={() => setType(t)} className={clsx("rounded-md px-2.5 py-1 text-xs uppercase", t === type ? "bg-emerald-500/20 text-emerald-200" : "text-slate-400 hover:bg-white/5")}>{t}</button>)}
      </div>
      <p className="mb-3 text-sm text-slate-400">{DESC[type]}</p>
      <div className="flex flex-wrap items-start gap-6">
        <T t={left} active={cur?.l} side="l" />
        <T t={right} active={cur?.r} side="r" />
        <div className="min-w-56 flex-1">
          <div className="mb-1 text-xs text-slate-400">result ({result.length} rows)</div>
          <table className="font-mono text-[12px]"><thead><tr>{[...left.columns, ...right.columns].map((c, i) => <th key={i} className="border-b border-white/10 px-2 py-1 text-left">{c}</th>)}</tr></thead>
            <tbody>{result.map((r, i) => <tr key={i} className={clsx(i === result.length - 1 && cur?.emit && "bg-emerald-500/15")}>{r.map((v, j) => <td key={j} className="px-2 py-0.5">{v === null ? <i className="text-amber-300/80">NULL</i> : v}</td>)}</tr>)}</tbody></table>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <button className="btn-primary text-xs" onClick={() => setPlaying(!playing)}>{playing ? <Pause size={12} /> : <Play size={12} />}{playing ? "Pause" : "Play"}</button>
        <button className="btn-outline text-xs" onClick={() => setStep((s) => Math.min(events.length, s + 1))}><SkipForward size={12} />Step</button>
        <button className="btn-ghost text-xs" onClick={() => { setStep(0); setPlaying(false); }}><RotateCcw size={12} />Reset</button>
        <span className="text-xs text-slate-500">{step}/{events.length} comparisons{cur ? (cur.l === null || cur.r === null ? " · emitting NULL-extended row" : cur.match ? " · keys match → emit" : " · no match") : ""}</span>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">This animates the logical result with a simple nested loop. MySQL may physically use index lookups or a hash join (8.0.18+) instead — the result is the same; check EXPLAIN for the real algorithm.</p>
    </div>
  );
}
