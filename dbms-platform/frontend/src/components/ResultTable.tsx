import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Clock, Hourglass, Info, XCircle } from "lucide-react";
import type { StatementResult } from "../lib/types";
import PlanTree, { isExplainJson, isExplainTree } from "./PlanTree";

function Cell({ v }: { v: string | number | null }) {
  if (v === null) return <span className="italic text-slate-500">NULL</span>;
  return <>{String(v)}</>;
}

export function ResultGrid({ st, maxHeight = 360 }: { st: StatementResult; maxHeight?: number }) {
  if (isExplainJson(st)) return <PlanTree json={String(st.rows[0][0])} />;
  if (isExplainTree(st)) {
    return <pre className="overflow-x-auto rounded-lg bg-ink-900 p-3 font-mono text-[12.5px] leading-relaxed text-emerald-200 scrollbar-thin">{String(st.rows[0][0])}</pre>;
  }
  return (
    <div className="overflow-auto rounded-lg border border-white/5 scrollbar-thin" style={{ maxHeight }}>
      <table className="min-w-full text-left font-mono text-[12.5px]">
        <thead className="sticky top-0 bg-ink-800">
          <tr>{st.columns.map((c, i) => <th key={i} className="whitespace-nowrap border-b border-white/10 px-3 py-1.5 font-semibold text-slate-300">{c}</th>)}</tr>
        </thead>
        <tbody>
          {st.rows.map((r, i) => (
            <tr key={i} className="odd:bg-white/[0.015] hover:bg-sky-500/5">
              {r.map((v, j) => <td key={j} className="whitespace-pre border-b border-white/5 px-3 py-1 align-top text-slate-200"><Cell v={v} /></td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StatementView({ st, showSql = false, compact = false }: { st: StatementResult; showSql?: boolean; compact?: boolean }) {
  const ok = st.status === "ok";
  return (
    <div className={clsx("space-y-2", !compact && "py-1")}>
      {showSql && <pre className="truncate font-mono text-[11.5px] text-slate-500">{st.sql.split("\n")[0]}{st.sql.includes("\n") ? " …" : ""}</pre>}
      {st.status === "blocked" && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
          <Hourglass size={15} className="animate-pulse" /> Waiting for a lock held by another session…
        </div>
      )}
      {(st.status === "error" || st.status === "rejected") && st.error && (
        <div className="rounded-lg border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-sm">
          <div className="flex items-start gap-2 text-rose-200">
            <XCircle size={15} className="mt-0.5 shrink-0" />
            <span className="font-mono text-[12.5px]">{st.status === "rejected" ? "Blocked by sandbox: " : `ERROR ${st.error.code ?? ""}: `}{st.error.message}</span>
          </div>
          {st.error.hint && <div className="mt-1.5 flex gap-2 pl-6 text-[13px] text-rose-100/80"><Info size={14} className="mt-0.5 shrink-0" />{st.error.hint}</div>}
        </div>
      )}
      {ok && st.columns.length > 0 && <ResultGrid st={st} />}
      {ok && st.more_results?.map((r, k) => (
        <div key={k} className="space-y-1">
          <div className="text-[10.5px] uppercase tracking-wider text-slate-500">Result set {k + 2}</div>
          <ResultGrid st={{ ...st, ...r, index: st.index }} />
        </div>
      ))}
      {ok && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-slate-500">
          <span className="inline-flex items-center gap-1 text-emerald-400/80"><CheckCircle2 size={12} />OK</span>
          {st.columns.length > 0 && <span>{st.row_count} row{st.row_count === 1 ? "" : "s"}{st.truncated ? " (display truncated)" : ""}</span>}
          {st.columns.length === 0 && st.affected_rows != null && <span>{st.affected_rows} row(s) affected</span>}
          {st.info && <span>{st.info}</span>}
          {st.elapsed_ms != null && <span className="inline-flex items-center gap-1"><Clock size={11} />{st.elapsed_ms} ms</span>}
        </div>
      )}
      {st.warnings && st.warnings.length > 0 && (
        <ul className="space-y-0.5 text-[12px] text-amber-300/90">
          {st.warnings.map((w, i) => <li key={i} className="flex gap-1.5"><AlertTriangle size={12} className="mt-0.5 shrink-0" />{w.level} {w.code}: {w.message}</li>)}
        </ul>
      )}
    </div>
  );
}

export function StatementList({ statements, showSql }: { statements: StatementResult[]; showSql?: boolean }) {
  const multi = statements.length > 1;
  return (
    <div className="divide-y divide-white/5">
      {statements.map((st) => <StatementView key={st.index + st.sql} st={st} showSql={showSql ?? multi} />)}
    </div>
  );
}
