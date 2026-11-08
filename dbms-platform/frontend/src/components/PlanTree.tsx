import clsx from "clsx";
import type { StatementResult } from "../lib/types";

export const isExplainJson = (st: StatementResult) =>
  st.columns.length === 1 && st.columns[0] === "EXPLAIN" && st.rows.length === 1 && String(st.rows[0][0]).trimStart().startsWith("{");
export const isExplainTree = (st: StatementResult) =>
  st.columns.length === 1 && st.columns[0] === "EXPLAIN" && st.rows.length === 1 && String(st.rows[0][0]).trimStart().startsWith("->");

interface PNode { label: string; detail: string[]; cost?: number; rows?: number; access?: string; children: PNode[] }

const BAD = new Set(["ALL"]);
const GOOD = new Set(["const", "eq_ref", "system"]);

function tableNode(t: Record<string, any>): PNode {
  const detail: string[] = [];
  if (t.key) detail.push(`key: ${t.key}${t.used_key_parts ? ` (${t.used_key_parts.join(", ")})` : ""}`);
  if (t.possible_keys) detail.push(`possible: ${t.possible_keys.join(", ")}`);
  if (t.attached_condition) detail.push(`filter: ${t.attached_condition}`);
  if (t.using_index) detail.push("covering index (Using index)");
  if (t.index_condition) detail.push(`ICP: ${t.index_condition}`);
  if (t.filtered) detail.push(`filtered: ${t.filtered}%`);
  const children: PNode[] = [];
  for (const k of ["materialized_from_subquery", "attached_subqueries"]) {
    const v = t[k];
    if (v) (Array.isArray(v) ? v : [v]).forEach((s: any) => children.push(walk(s.query_block ?? s)));
  }
  return {
    label: `${t.access_type ?? "?"} on ${t.table_name}`, access: t.access_type, detail, children,
    rows: t.rows_examined_per_scan, cost: t.cost_info ? Number(t.cost_info.prefix_cost ?? t.cost_info.read_cost) : undefined,
  };
}

function walk(o: Record<string, any>): PNode {
  if (o.table) return tableNode(o.table);
  if (o.query_block) return walk(o.query_block);
  const children: PNode[] = [];
  let label = "query block";
  const detail: string[] = [];
  if (o.select_id) label = `query block #${o.select_id}`;
  if (o.message) detail.push(o.message);
  for (const k of ["ordering_operation", "grouping_operation", "duplicates_removal", "windowing"]) {
    if (o[k]) {
      const inner = walk(o[k]);
      const flags = [o[k].using_filesort && "filesort", o[k].using_temporary_table && "temporary table"].filter(Boolean);
      return { label: k.replace("_", " "), detail: [...detail, ...flags.map((f) => `uses ${f}`)], children: [inner], cost: o.cost_info ? Number(o.cost_info.query_cost) : undefined };
    }
  }
  if (o.nested_loop) { label = "nested loop join"; o.nested_loop.forEach((n: any) => children.push(walk(n))); }
  if (o.union_result) { label = "union"; (o.union_result.query_specifications ?? []).forEach((q: any) => children.push(walk(q))); }
  if (o.table) children.push(tableNode(o.table));
  return { label, detail, children, cost: o.cost_info ? Number(o.cost_info.query_cost) : undefined };
}

function NodeView({ n, depth }: { n: PNode; depth: number }) {
  return (
    <div className={clsx(depth > 0 && "ml-5 border-l border-white/10 pl-4")}>
      <div className={clsx("my-1.5 rounded-lg border px-3 py-2",
        n.access && BAD.has(n.access) ? "border-rose-400/40 bg-rose-500/10" : n.access && GOOD.has(n.access) ? "border-emerald-400/40 bg-emerald-500/10" : "border-sky-400/25 bg-sky-500/5")}>
        <div className="flex flex-wrap items-baseline gap-x-3 font-mono text-[12.5px]">
          <span className="font-semibold text-slate-100">{n.label}</span>
          {n.rows != null && <span className="text-slate-400">rows≈{n.rows}</span>}
          {n.cost != null && !Number.isNaN(n.cost) && <span className="text-slate-500">cost {n.cost}</span>}
          {n.access === "ALL" && <span className="chip bg-rose-500/20 text-rose-200">full table scan</span>}
        </div>
        {n.detail.map((d, i) => <div key={i} className="mt-0.5 break-all font-mono text-[11.5px] text-slate-400">{d}</div>)}
      </div>
      {n.children.map((c, i) => <NodeView key={i} n={c} depth={depth + 1} />)}
    </div>
  );
}

export default function PlanTree({ json }: { json: string }) {
  let root: PNode;
  try { root = walk(JSON.parse(json)); } catch { return <pre className="text-xs">{json}</pre>; }
  return (
    <div className="rounded-lg bg-ink-900/70 p-3">
      <div className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">Execution plan (EXPLAIN FORMAT=JSON) — green: single-row access · red: full scan</div>
      <NodeView n={root} depth={0} />
      <details className="mt-2 text-xs text-slate-500"><summary className="cursor-pointer">Raw JSON</summary><pre className="mt-1 max-h-72 overflow-auto scrollbar-thin">{json}</pre></details>
    </div>
  );
}
