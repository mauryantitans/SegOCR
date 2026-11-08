import { Play, RotateCcw, Search, SkipForward } from "lucide-react";
import { useMemo, useState } from "react";
import { BPlusTree, type Snap } from "./bptree";

interface Layout { node: Snap; x: number; y: number; w: number }
const KEY_W = 30, PAD = 10, H = 30, VGAP = 70, HGAP = 14;

function layout(root: Snap) {
  const out: Layout[] = []; const edges: [Layout, Layout][] = [];
  let cursor = 0;
  const place = (n: Snap, depth: number): Layout => {
    const w = Math.max(1, n.keys.length) * KEY_W + PAD;
    if (n.leaf) { const l = { node: n, x: cursor, y: depth * VGAP, w }; cursor += w + HGAP; out.push(l); return l; }
    const kids = n.children.map((c) => place(c, depth + 1));
    const cx = (kids[0].x + kids[kids.length - 1].x + kids[kids.length - 1].w) / 2;
    const l = { node: n, x: cx - w / 2, y: depth * VGAP, w };
    out.push(l); kids.forEach((k) => edges.push([l, k]));
    return l;
  };
  place(root, 0);
  const minX = Math.min(...out.map((o) => o.x));
  out.forEach((o) => (o.x -= minX - 10));
  const width = Math.max(...out.map((o) => o.x + o.w)) + 10;
  const height = Math.max(...out.map((o) => o.y)) + H + 10;
  return { nodes: out, edges, width, height };
}

export default function BPlusTreeViz({ order = 4, keys = [10, 20, 5, 6, 12, 30, 7, 17], search }: { order?: number; keys?: number[]; search?: number }) {
  const [ord, setOrd] = useState(order);
  const [inserted, setInserted] = useState<number[]>(keys);
  const [pending, setPending] = useState<number[]>([]);
  const [input, setInput] = useState("");
  const [q, setQ] = useState(search != null ? String(search) : "");
  const [range, setRange] = useState({ lo: "", hi: "" });
  const [hl, setHl] = useState<{ path: number[]; leaves?: number[]; keys?: number[] } | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const tree = useMemo(() => { const t = new BPlusTree(ord); inserted.forEach((k) => t.insert(k)); return t; }, [ord, inserted]);
  const snap = tree.snapshot();
  const L = layout(snap);

  const insert = (k: number) => {
    const t = new BPlusTree(ord); inserted.forEach((x) => t.insert(x));
    const ev = t.insert(k);
    setInserted((s) => (s.includes(k) ? s : [...s, k])); setLog(ev); setHl(null);
  };
  const doSearch = () => { const k = Number(q); if (Number.isNaN(k)) return; const r = tree.search(k); setHl({ path: r.path, keys: r.found ? [k] : [] }); setLog(r.steps); };
  const doRange = () => {
    const lo = Number(range.lo), hi = Number(range.hi); if (Number.isNaN(lo) || Number.isNaN(hi)) return;
    const r = tree.range(lo, hi); setHl({ path: r.path, leaves: r.leaves, keys: r.keys });
    setLog([`Descend once to the leaf containing ${lo} (${r.path.length} node reads), then follow leaf links across ${r.leaves.length} leaf page(s).`, `Matching keys: ${r.keys.join(", ") || "none"}`]);
  };
  const replay = () => { setPending(inserted); setInserted([]); setLog(["Replaying inserts one at a time — press Step."]); setHl(null); };
  const step = () => { if (!pending.length) return; const [k, ...rest] = pending; setPending(rest); insert(k); };

  return (
    <div className="not-prose my-5 rounded-xl border border-pink-400/20 bg-ink-900/60 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="font-semibold uppercase tracking-wider text-pink-300">B+ tree</span>
        <label className="ml-2 text-slate-400">order
          <select className="ml-1 rounded bg-ink-800 px-1 py-0.5" value={ord} onChange={(e) => { setOrd(Number(e.target.value)); setHl(null); setLog([]); }}>
            {[3, 4, 5, 6, 7].map((o) => <option key={o}>{o}</option>)}
          </select></label>
        <span className="text-slate-500">max {ord - 1} keys/node · height {tree.height()} · {inserted.length} keys</span>
      </div>
      <div className="mb-3 flex flex-wrap gap-2">
        <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); const k = Number(input); if (!Number.isNaN(k) && input) { insert(k); setInput(""); } }}>
          <input className="input w-24 py-1" placeholder="key" value={input} onChange={(e) => setInput(e.target.value)} /><button className="btn-primary text-xs">Insert</button></form>
        <div className="flex gap-1"><input className="input w-20 py-1" placeholder="find" value={q} onChange={(e) => setQ(e.target.value)} /><button className="btn-outline text-xs" onClick={doSearch}><Search size={12} />Search</button></div>
        <div className="flex gap-1"><input className="input w-16 py-1" placeholder="from" value={range.lo} onChange={(e) => setRange({ ...range, lo: e.target.value })} /><input className="input w-16 py-1" placeholder="to" value={range.hi} onChange={(e) => setRange({ ...range, hi: e.target.value })} /><button className="btn-outline text-xs" onClick={doRange}>Range scan</button></div>
        <button className="btn-ghost text-xs" onClick={replay}><Play size={12} />Replay inserts</button>
        {pending.length > 0 && <button className="btn-primary text-xs" onClick={step}><SkipForward size={12} />Step ({pending.length} left)</button>}
        <button className="btn-ghost text-xs" onClick={() => { setInserted([]); setPending([]); setHl(null); setLog([]); }}><RotateCcw size={12} />Clear</button>
      </div>
      <div className="overflow-x-auto rounded-lg bg-ink-950/60 p-2 scrollbar-thin">
        <svg width={L.width} height={L.height} className="mx-auto block">
          {L.edges.map(([a, b], i) => <line key={i} x1={a.x + a.w / 2} y1={a.y + H} x2={b.x + b.w / 2} y2={b.y} stroke={hl?.path.includes(b.node.id) && hl.path.includes(a.node.id) ? "#f472b6" : "#334155"} strokeWidth={1.5} />)}
          {L.nodes.filter((n) => n.node.leaf).map((n, i, arr) => i < arr.length - 1 ? <line key={"l" + i} x1={n.x + n.w} y1={n.y + H / 2} x2={arr[i + 1].x} y2={arr[i + 1].y + H / 2} stroke="#475569" strokeDasharray="3 3" markerEnd="url(#arr)" /> : null)}
          <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#475569" /></marker></defs>
          {L.nodes.map((n) => {
            const onPath = hl?.path.includes(n.node.id) || hl?.leaves?.includes(n.node.id);
            return (
              <g key={n.node.id} transform={`translate(${n.x},${n.y})`}>
                <rect width={n.w} height={H} rx={6} fill={onPath ? "rgba(244,114,182,0.18)" : n.node.leaf ? "rgba(52,211,153,0.08)" : "rgba(56,189,248,0.08)"} stroke={onPath ? "#f472b6" : n.node.leaf ? "#34d39966" : "#38bdf866"} />
                {n.node.keys.map((k, i) => (
                  <text key={i} x={PAD / 2 + i * KEY_W + KEY_W / 2} y={H / 2 + 4} textAnchor="middle" fontSize={12} fontFamily="JetBrains Mono, monospace" fill={hl?.keys?.includes(k) && n.node.leaf ? "#fde68a" : "#e2e8f0"} fontWeight={hl?.keys?.includes(k) && n.node.leaf ? 700 : 400}>{k}</text>
                ))}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex gap-4 text-[11px] text-slate-500"><span><span className="text-sky-300">■</span> internal nodes (separator keys only)</span><span><span className="text-emerald-300">■</span> leaves (all keys, linked for range scans)</span></div>
      {log.length > 0 && <ul className="mt-2 space-y-0.5 font-mono text-[12px] text-slate-300">{log.map((l, i) => <li key={i}>› {l}</li>)}</ul>}
    </div>
  );
}
