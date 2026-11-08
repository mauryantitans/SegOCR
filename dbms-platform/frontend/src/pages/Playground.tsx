import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronDown, ChevronRight, GitFork, History, KeyRound, Loader2, Play, RotateCcw, Table2 } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import PlanTree from "../components/PlanTree";
import { StatementList } from "../components/ResultTable";
import { api } from "../lib/api";
import { useDatasets } from "../lib/queries";
import type { RunResult } from "../lib/types";

const SqlEditor = lazy(() => import("../components/SqlEditor"));
const HISTORY_KEY = "dbacademy.history";
const DEFAULT_SQL = `-- Your private MySQL lab. Ctrl+Enter runs the editor (or just the selection).
SELECT c.first_name, c.last_name, COUNT(o.order_id) AS orders
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY orders DESC, c.customer_id
LIMIT 10;`;

interface SchemaInfo {
  schema: string; dataset: string; routines: { name: string; type: string }[];
  tables: { name: string; type: string; approxRows: number; columns: { name: string; type: string; nullable: boolean; key: string; extra: string }[];
    indexes: { name: string; unique: boolean; columns: string[] }[]; foreignKeys: { name: string; columns: string[]; refTable: string; refColumns: string[] }[] }[];
}

function loadHistory(): string[] { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); } catch { return []; } }

export default function Playground() {
  const loc = useLocation() as { state?: { sql?: string; dataset?: string } };
  const qc = useQueryClient();
  const { data: datasets } = useDatasets();
  const [dataset, setDataset] = useState(loc.state?.dataset ?? "shop");
  const [sql, setSql] = useState(loc.state?.sql ?? DEFAULT_SQL);
  const [result, setResult] = useState<RunResult | null>(null);
  const [plan, setPlan] = useState<{ json: string; tree: string | null } | null>(null);
  const [busy, setBusy] = useState<"" | "run" | "explain" | "reset">("");
  const [err, setErr] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>(loadHistory);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => { if (loc.state?.sql) { setSql(loc.state.sql); if (loc.state.dataset) setDataset(loc.state.dataset); } }, [loc.state]);
  const { data: schema, refetch } = useQuery({
    queryKey: ["schema", "playground", dataset],
    queryFn: () => api<SchemaInfo>(`/labs/schema?lab=playground&dataset=${dataset}`),
  });
  useEffect(() => {
    if (schema) import("../components/SqlEditor").then((m) => m.setEditorSchemaWords(schema.tables.flatMap((t) => [t.name, ...t.columns.map((c) => c.name)])));
  }, [schema]);

  const run = async (text: string) => {
    setBusy("run"); setErr(null); setPlan(null);
    try {
      const r = await api<RunResult>("/labs/run", { method: "POST", json: { lab: "playground", sql: text, dataset, stop_on_error: true } });
      setResult(r);
      const h = [text, ...history.filter((x) => x !== text)].slice(0, 30);
      setHistory(h); try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h)); } catch { /* ignore */ }
      if (r.statements.some((s) => s.kind === "ddl" || s.kind === "write" || s.kind === "routine")) refetch();
    } catch (e) { setErr((e as Error).message); } finally { setBusy(""); }
  };
  const explain = async () => {
    setBusy("explain"); setErr(null);
    try { setPlan(await api("/labs/explain", { method: "POST", json: { lab: "playground", sql, dataset } })); setResult(null); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(""); }
  };
  const reset = async (ds = dataset) => {
    setBusy("reset"); setErr(null);
    try { await api("/labs/reset", { method: "POST", json: { lab: "playground", dataset: ds } }); setResult(null); setPlan(null); qc.invalidateQueries({ queryKey: ["schema", "playground"] }); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(""); }
  };

  return (
    <div className="flex h-full flex-col gap-4 p-4 lg:flex-row lg:p-6">
      <aside className="glass w-full shrink-0 overflow-y-auto p-3 scrollbar-thin lg:w-72">
        <label className="block text-xs text-slate-400">Dataset
          <select className="input mt-1 py-1.5" value={dataset} onChange={(e) => { setDataset(e.target.value); reset(e.target.value); }}>
            {Object.entries(datasets ?? {}).map(([k, v]) => <option key={k} value={k}>{v.title}</option>)}
          </select></label>
        {datasets?.[dataset] && <p className="mt-2 text-[11.5px] leading-relaxed text-slate-500">{datasets[dataset].description}</p>}
        <div className="mt-3 flex items-center justify-between text-xs text-slate-400"><span>Tables & views</span>
          <button className="btn-ghost px-1.5 py-0.5 text-[11px]" onClick={() => reset()} disabled={!!busy}><RotateCcw size={11} />Reset lab</button></div>
        <div className="mt-1 space-y-0.5">
          {schema?.tables.map((t) => (
            <div key={t.name}>
              <button className="flex w-full items-center gap-1.5 rounded px-1.5 py-1 text-left text-[13px] hover:bg-white/5" onClick={() => setOpen((o) => ({ ...o, [t.name]: !o[t.name] }))}>
                {open[t.name] ? <ChevronDown size={13} /> : <ChevronRight size={13} />}<Table2 size={13} className={t.type === "view" ? "text-violet-300" : "text-sky-300"} />
                <span className="font-mono text-slate-200">{t.name}</span><span className="ml-auto text-[10px] text-slate-600">{t.type === "view" ? "view" : `~${t.approxRows ?? 0}`}</span>
              </button>
              {open[t.name] && (
                <div className="mb-1 ml-6 space-y-0.5 border-l border-white/5 pl-2 font-mono text-[11.5px]">
                  {t.columns.map((c) => (
                    <div key={c.name} className="flex items-center gap-1 text-slate-400" title={c.extra}>
                      {c.key === "PRI" ? <KeyRound size={10} className="text-amber-300" /> : c.key ? <KeyRound size={10} className="text-slate-500" /> : <span className="w-2.5" />}
                      <span className="text-slate-300">{c.name}</span><span className="truncate text-slate-600">{c.type}{c.nullable ? "" : " NN"}</span>
                    </div>
                  ))}
                  {t.indexes.length > 0 && <div className="pt-1 text-[10.5px] text-slate-500">indexes: {t.indexes.map((i) => `${i.name}(${i.columns.join(",")})`).join(" · ")}</div>}
                  {t.foreignKeys.map((f) => <div key={f.name} className="flex items-center gap-1 text-[10.5px] text-slate-500"><GitFork size={10} />{f.columns.join(",")} → {f.refTable}({f.refColumns.join(",")})</div>)}
                </div>
              )}
            </div>
          ))}
          {schema && !schema.tables.length && <div className="px-2 py-1 text-xs text-slate-500">No tables yet — create some!</div>}
          {!!schema?.routines.length && <div className="pt-2 text-[11px] text-slate-500">Routines: {schema.routines.map((r) => `${r.name} (${r.type.toLowerCase()})`).join(", ")}</div>}
        </div>
      </aside>
      <section className="flex min-w-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="mr-auto text-lg font-semibold text-white">SQL Playground</h1>
          <button className="btn-ghost text-xs" onClick={() => setShowHistory(!showHistory)}><History size={13} />History</button>
          <button className="btn-outline text-xs" onClick={explain} disabled={!!busy}>{busy === "explain" ? <Loader2 size={13} className="animate-spin" /> : <GitFork size={13} />}Visual EXPLAIN</button>
          <button className="btn-primary text-xs" onClick={() => run(sql)} disabled={!!busy}>{busy === "run" ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}Run (Ctrl+Enter)</button>
        </div>
        {showHistory && (
          <div className="glass max-h-48 overflow-y-auto p-2 scrollbar-thin">
            {history.length ? history.map((h, i) => <button key={i} className="block w-full truncate rounded px-2 py-1 text-left font-mono text-xs text-slate-400 hover:bg-white/5 hover:text-slate-200" onClick={() => { setSql(h); setShowHistory(false); }}>{h.replace(/\s+/g, " ")}</button>) : <div className="p-2 text-xs text-slate-500">No history yet.</div>}
          </div>
        )}
        <Suspense fallback={<div className="h-64 rounded-xl bg-ink-900" />}><SqlEditor value={sql} onChange={setSql} onRun={run} height={280} /></Suspense>
        <div className="glass min-h-40 flex-1 overflow-auto p-3 scrollbar-thin">
          {err && <div className="mb-2 rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{err}</div>}
          {plan && <>{plan.tree && <pre className="mb-3 overflow-x-auto rounded-lg bg-ink-900 p-3 font-mono text-[12.5px] text-emerald-200">{plan.tree}</pre>}<PlanTree json={plan.json} /></>}
          {result && (
            <>
              <div className="mb-2 flex flex-wrap gap-3 text-[11px] text-slate-500"><span>MySQL {result.server_version}</span><span>dataset: {result.dataset}</span><span>{result.statements.length} statement(s) · {result.elapsed_ms} ms</span></div>
              {result.notice && <div className="mb-2 text-xs text-amber-300">{result.notice}</div>}
              <StatementList statements={result.statements} />
            </>
          )}
          {!result && !plan && !err && (
            <div className={clsx("text-sm text-slate-500")}>
              Results appear here. Your lab is a private MySQL schema loaded with the <b>{dataset}</b> dataset — create tables, modify data, break things; <b>Reset lab</b> restores it.
              Statements run with a {10}-second limit; server administration (users, GRANT, global variables, files) is disabled.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
