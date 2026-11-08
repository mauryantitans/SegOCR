import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { ExternalLink, History, Loader2, Play, RotateCcw } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../lib/api";
import type { Capture, RunResult, StatementResult } from "../../lib/types";
import { CodeBlock } from "../Markdown";
import { StatementList } from "../ResultTable";
import { useLessonCtx } from "./context";

export function CapturedOutput({ capture, measured, version }: { capture?: Capture | null; measured?: boolean; version?: string }) {
  if (!capture?.statements) return null;
  const sts = capture.statements.filter((s) => s.status !== "ok" || s.columns?.length || s.affected_rows != null || s.info || s.warnings?.length) as StatementResult[];
  if (!sts.length) return <div className="px-1 text-[11.5px] text-slate-500">Statement(s) executed successfully (no result set).</div>;
  return (
    <div className="space-y-1">
      <div className="text-[10.5px] uppercase tracking-wider text-slate-500">
        {measured ? `Measured on the build machine${version ? ` (MySQL ${version})` : ""} — your timings will differ` : `Output captured from MySQL${version ? ` ${version}` : ""}`}
      </div>
      <StatementList statements={sts.map((s, i) => ({ ...s, index: i, columns: s.columns ?? [], rows: s.rows ?? [] }))} showSql={sts.length > 1} />
    </div>
  );
}

export function useLessonRunner() {
  const ctx = useLessonCtx();
  const lab = `lesson:${ctx.docId}`;
  const run = (sql: string) => api<RunResult>("/labs/run", { method: "POST", json: { lab, sql, stop_on_error: true } });
  const replayUntil = async (blockId: string) => {
    await api("/labs/reset", { method: "POST", json: { lab } });
    let last: RunResult | null = null;
    for (const b of ctx.runnable) {
      last = await api<RunResult>("/labs/run", { method: "POST", json: { lab, sql: b.sql, stop_on_error: false } });
      if (b.id === blockId) break;
    }
    return last!;
  };
  return { run, replayUntil, lab };
}

export default function SqlBlock({ id, sql, run: runnable, expect, volatile, measured, capture }: {
  id: string; sql: string; run: boolean; expect: "ok" | "error"; volatile: boolean; measured: boolean; capture?: Capture | null;
}) {
  const ctx = useLessonCtx();
  const nav = useNavigate();
  const { run, replayUntil } = useLessonRunner();
  const [live, setLive] = useState<RunResult | null>(null);
  const exec = useMutation({ mutationFn: () => run(sql), onSuccess: setLive });
  const replay = useMutation({ mutationFn: () => replayUntil(id), onSuccess: setLive });
  const busy = exec.isPending || replay.isPending;
  const missingTable = live?.statements.some((s) => s.error?.code === 1146 || s.error?.code === 1054);

  return (
    <div className={clsx("not-prose my-5 overflow-hidden rounded-xl border bg-ink-900/60", expect === "error" ? "border-rose-400/20" : "border-white/[0.07]")}>
      <div className="flex items-center justify-between gap-2 border-b border-white/5 bg-white/[0.02] px-3 py-1.5">
        <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
          {runnable ? (expect === "error" ? "SQL · expected to fail" : "SQL · runnable") : "SQL"}
        </span>
        <div className="flex gap-1">
          <button className="btn-ghost px-2 py-1 text-xs" onClick={() => nav("/playground", { state: { sql, dataset: ctx.dataset } })} title="Open in playground"><ExternalLink size={13} />Playground</button>
          {runnable && <button className="btn-ghost px-2 py-1 text-xs" onClick={() => replay.mutate()} disabled={busy} title="Reset this lesson's lab and run every block up to this one"><History size={13} />Replay to here</button>}
          {runnable && <button className="btn-primary px-2.5 py-1 text-xs" onClick={() => exec.mutate()} disabled={busy}>{busy ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}Run</button>}
        </div>
      </div>
      <div className="[&_pre]:my-0 [&_pre]:rounded-none [&_pre]:border-0 [&_pre]:bg-transparent"><CodeBlock code={sql} lang="sql" /></div>
      {(live || (capture && !volatile) || exec.error || replay.error) && (
        <div className="border-t border-white/5 px-3 py-3">
          {(exec.error || replay.error) && <div className="text-sm text-rose-300">{String((exec.error ?? replay.error as Error).message)}</div>}
          {live ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[10.5px] uppercase tracking-wider text-sky-300/80">
                <span>Live result from your lab · MySQL {live.server_version}</span>
                <button className="normal-case text-slate-500 hover:text-slate-300" onClick={() => setLive(null)}><RotateCcw size={11} className="mr-1 inline" />show captured</button>
              </div>
              <StatementList statements={live.statements} />
              {missingTable && <div className="rounded-lg bg-sky-500/10 px-3 py-2 text-xs text-sky-200">This block depends on objects created by earlier blocks. Click <b>Replay to here</b> to rebuild your lab up to this point.</div>}
              {live.notice && <div className="text-xs text-amber-300">{live.notice}</div>}
            </div>
          ) : !volatile ? <CapturedOutput capture={capture} measured={measured} version={ctx.captureVersion} /> : null}
        </div>
      )}
      {volatile && !live && <div className="border-t border-white/5 px-3 py-2 text-[11.5px] text-slate-500">Output varies between runs — press Run to see your own result.</div>}
    </div>
  );
}
