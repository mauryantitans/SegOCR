import clsx from "clsx";
import { RotateCcw, SkipForward } from "lucide-react";
import { useMemo, useState } from "react";

/** InnoDB midpoint-insertion LRU (innodb_old_blocks_pct, default 37). innodb_old_blocks_time is ignored for simplicity. */
function simulate(capacity: number, accesses: (string | number)[], oldPct: number) {
  const states: { list: string[]; boundary: number; event: string; hit: boolean; page: string }[] = [];
  let list: string[] = [];
  const oldLen = () => Math.max(1, Math.round((list.length * oldPct) / 100));
  for (const raw of accesses) {
    const p = String(raw);
    const idx = list.indexOf(p);
    let event: string; let hit = false;
    if (idx >= 0) {
      hit = true;
      const youngLen = list.length - oldLen();
      if (idx >= youngLen) { list.splice(idx, 1); list.unshift(p); event = `hit in OLD sublist → page ${p} made young (moved to head)`; }
      else event = `hit in YOUNG sublist → page ${p} stays (InnoDB avoids churning the hot head)`;
    } else {
      let evicted = "";
      if (list.length >= capacity) evicted = list.pop()!;
      const youngLen = list.length - (list.length ? Math.max(1, Math.round((list.length * oldPct) / 100)) : 0);
      list.splice(Math.max(0, youngLen), 0, p);
      event = `miss → read page ${p} from disk, insert at the MIDPOINT (head of old sublist)${evicted ? `; evict ${evicted} from the tail` : ""}`;
    }
    list = [...list];
    states.push({ list, boundary: list.length - oldLen(), event, hit, page: p });
  }
  return states;
}

export default function BufferPoolViz({ capacity = 8, accesses = [1, 2, 3, 4, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2], oldPct = 37 }: { capacity?: number; accesses?: (string | number)[]; oldPct?: number }) {
  const [seq, setSeq] = useState(accesses.join(" "));
  const parsed = seq.split(/[\s,]+/).filter(Boolean);
  const states = useMemo(() => simulate(capacity, parsed, oldPct), [capacity, seq, oldPct]); // eslint-disable-line react-hooks/exhaustive-deps
  const [i, setI] = useState(0);
  const st = states[Math.min(i, states.length) - 1];
  const hits = states.slice(0, i).filter((s) => s.hit).length;
  return (
    <div className="not-prose my-5 rounded-xl border border-pink-400/20 bg-ink-900/60 p-4 text-sm">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-pink-300">InnoDB buffer pool LRU (midpoint insertion)</div>
      <label className="block text-xs text-slate-400">Page access sequence<input className="input mt-1 font-mono" value={seq} onChange={(e) => { setSeq(e.target.value); setI(0); }} /></label>
      <div className="mt-3 flex items-center gap-2">
        <button className="btn-primary text-xs" onClick={() => setI((x) => Math.min(states.length, x + 1))} disabled={i >= states.length}><SkipForward size={12} />Access next page</button>
        <button className="btn-outline text-xs" onClick={() => setI(states.length)}>Run all</button>
        <button className="btn-ghost text-xs" onClick={() => setI(0)}><RotateCcw size={12} />Reset</button>
        <span className="text-xs text-slate-500">{i}/{states.length} accesses · hits {hits} · misses {i - hits} · capacity {capacity} pages · old = {oldPct}%</span>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-1">
        <span className="mr-1 text-[11px] text-slate-500">head (MRU)</span>
        {(st?.list ?? []).map((p, k) => (
          <span key={p} className={clsx("grid h-9 w-9 place-items-center rounded-md border font-mono text-xs",
            k >= (st?.boundary ?? 0) ? "border-slate-500/50 bg-slate-500/10 text-slate-300" : "border-pink-400/50 bg-pink-500/15 text-pink-100",
            p === st?.page && "ring-2 ring-amber-300")}>{p}</span>
        ))}
        <span className="ml-1 text-[11px] text-slate-500">tail (evicted next)</span>
      </div>
      <div className="mt-1 flex gap-4 text-[11px] text-slate-500"><span><span className="text-pink-300">■</span> young sublist</span><span><span className="text-slate-400">■</span> old sublist</span></div>
      {st && <div className={clsx("mt-2 font-mono text-[12px]", st.hit ? "text-emerald-300" : "text-amber-200")}>› {st.event}</div>}
      <p className="mt-2 text-[11px] text-slate-500">Why midpoint insertion: a one-off full table scan reads many pages once; they enter the old sublist and age out without flushing the hot working set from the young sublist. Real InnoDB also requires a page to stay in the old sublist for innodb_old_blocks_time (default 1000 ms) before an access makes it young.</p>
    </div>
  );
}
