import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, KeyRound, Lightbulb, Loader2, Play, Send, Trophy, XCircle } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { ExerciseDef, RunResult } from "../../lib/types";
import Markdown, { CodeBlock } from "../Markdown";
import { ResultGrid, StatementList } from "../ResultTable";

const SqlEditor = lazy(() => import("../SqlEditor"));
const DIFF: Record<string, string> = { easy: "text-emerald-300 bg-emerald-500/10", medium: "text-sky-300 bg-sky-500/10", hard: "text-amber-300 bg-amber-500/10", challenge: "text-rose-300 bg-rose-500/10" };

interface Grade {
  passed: boolean; summary: string; details: string[]; attempts: number; solutionUnlocked: boolean;
  your_result?: { columns: string[]; rows: (string | number | null)[][]; row_count: number } | null;
  missing_rows?: string[][]; unexpected_rows?: string[][]; solution?: string; explanation?: string;
}

export default function ExerciseBlock(ex: ExerciseDef) {
  const qc = useQueryClient();
  const isOpen = ex.kind === "open";
  const { data: status } = useQuery({
    queryKey: ["exercise", ex.id],
    queryFn: () => api<{ attempts: number; passed: boolean; solutionUnlocked: boolean; lastSubmission: string | null }>(`/exercises/${ex.id}/status`),
  });
  const [code, setCode] = useState(ex.starter ?? (isOpen ? "" : "-- Write your SQL here\n"));
  useEffect(() => { if (status?.lastSubmission) setCode(status.lastSubmission); }, [status?.lastSubmission]);
  const [hints, setHints] = useState(0);
  const [grade, setGrade] = useState<Grade | null>(null);
  const [trial, setTrial] = useState<RunResult | null>(null);
  const [solution, setSolution] = useState<{ solution: string; explanation?: string } | null>(null);
  const [busy, setBusy] = useState<"" | "run" | "submit" | "solution">("");
  const [err, setErr] = useState<string | null>(null);

  const act = async (kind: "run" | "submit" | "solution") => {
    setBusy(kind); setErr(null);
    try {
      if (kind === "run") {
        setTrial(await api<RunResult>("/labs/run", { method: "POST", json: { lab: `exercise:${ex.id}`, sql: code, stop_on_error: true } }));
      } else if (kind === "submit") {
        const g = await api<Grade>(`/exercises/${ex.id}/attempt`, { method: "POST", json: { submission: code, hints_used: hints } });
        setGrade(g); setTrial(null);
        if (g.solution) setSolution({ solution: g.solution, explanation: g.explanation });
        qc.invalidateQueries({ queryKey: ["exercise", ex.id] }); qc.invalidateQueries({ queryKey: ["progress"] }); qc.invalidateQueries({ queryKey: ["weak"] });
      } else {
        setSolution(await api(`/exercises/${ex.id}/solution`));
      }
    } catch (e) { setErr((e as Error).message); } finally { setBusy(""); }
  };
  const unlocked = grade?.solutionUnlocked ?? status?.solutionUnlocked;
  const solved = grade?.passed ?? status?.passed;

  return (
    <div className={clsx("not-prose my-6 rounded-xl border bg-ink-900/50", solved ? "border-emerald-400/30" : "border-rose-300/15")}>
      <div className="flex flex-wrap items-center gap-2 border-b border-white/5 px-4 py-2.5">
        {solved ? <Trophy size={15} className="text-emerald-300" /> : <KeyRound size={15} className="text-rose-300" />}
        <span className="text-sm font-semibold text-slate-100">{ex.title}</span>
        {ex.difficulty && <span className={clsx("chip capitalize", DIFF[ex.difficulty])}>{ex.difficulty}</span>}
        {isOpen && <span className="chip bg-violet-500/10 text-violet-300">written answer</span>}
        {ex.dataset && !isOpen && <span className="chip bg-white/5 text-slate-400">dataset: {ex.dataset}</span>}
        {status && status.attempts > 0 && <span className="ml-auto text-[11px] text-slate-500">{status.attempts} attempt{status.attempts === 1 ? "" : "s"}</span>}
      </div>
      <div className="space-y-3 p-4">
        <Markdown content={ex.prompt} className="callout-body" />
        <Suspense fallback={<div className="h-40 rounded-xl bg-ink-900" />}>
          {isOpen
            ? <textarea className="input min-h-40 font-sans" placeholder="Write your answer…" value={code} onChange={(e) => setCode(e.target.value)} />
            : <SqlEditor value={code} onChange={setCode} height={170} onRun={() => act("run")} />}
        </Suspense>
        <div className="flex flex-wrap gap-2">
          {!isOpen && <button className="btn-outline text-xs" onClick={() => act("run")} disabled={!!busy}>{busy === "run" ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}Try it (Ctrl+Enter)</button>}
          <button className="btn-primary text-xs" onClick={() => act("submit")} disabled={!!busy || !code.trim()}>{busy === "submit" ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}{isOpen ? "Submit & compare" : "Submit for grading"}</button>
          {(ex.hints?.length ?? 0) > 0 && <button className="btn-ghost text-xs" onClick={() => setHints((h) => Math.min(h + 1, ex.hints!.length))} disabled={hints >= ex.hints!.length}><Lightbulb size={13} />{hints >= ex.hints!.length ? "No more hints" : `Hint ${hints + 1}/${ex.hints!.length}`}</button>}
          {!isOpen && !solution && <button className="btn-ghost text-xs" onClick={() => act("solution")} disabled={!unlocked || !!busy} title={unlocked ? "" : "Unlocks after 2 attempts or a correct answer"}>{unlocked ? "Show reference solution" : "Solution locked (2 attempts)"}</button>}
        </div>
        {hints > 0 && <ol className="space-y-1.5">{ex.hints!.slice(0, hints).map((h, i) => (
          <li key={i} className="flex gap-2 rounded-lg bg-amber-500/[0.07] px-3 py-2 text-sm text-amber-100"><Lightbulb size={14} className="mt-0.5 shrink-0 text-amber-300" /><Markdown content={h} className="callout-body" /></li>))}</ol>}
        {err && <div className="rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{err}</div>}
        {trial && <div className="rounded-lg border border-white/5 p-3"><div className="mb-1 text-[10.5px] uppercase tracking-wider text-slate-500">Trial run (not graded)</div><StatementList statements={trial.statements} /></div>}
        {grade && (
          <div className={clsx("rounded-lg border p-3", grade.passed ? "border-emerald-400/30 bg-emerald-500/[0.06]" : "border-rose-400/30 bg-rose-500/[0.06]")}>
            <div className={clsx("flex items-start gap-2 text-sm", grade.passed ? "text-emerald-200" : "text-rose-200")}>{grade.passed ? <CheckCircle2 size={16} className="mt-0.5 shrink-0" /> : <XCircle size={16} className="mt-0.5 shrink-0" />}{grade.summary}</div>
            {grade.details.map((d, i) => <div key={i} className="mt-1 pl-6 text-[13px] text-slate-300">{d}</div>)}
            {!!grade.unexpected_rows?.length && <div className="mt-2 pl-6 text-xs text-slate-400">Rows in your result that should not be there (first {grade.unexpected_rows.length}):<pre className="mt-1 font-mono text-rose-200/80">{grade.unexpected_rows.map((r) => r.join(" | ")).join("\n")}</pre></div>}
            {!!grade.missing_rows?.length && <div className="mt-2 pl-6 text-xs text-slate-400">Expected rows missing from your result:<pre className="mt-1 font-mono text-emerald-200/80">{grade.missing_rows.map((r) => r.join(" | ")).join("\n")}</pre></div>}
            {grade.your_result && !grade.passed && <details className="mt-2 pl-6 text-xs text-slate-500"><summary className="cursor-pointer">Your result</summary><div className="mt-1"><ResultGrid st={{ index: 0, sql: "", status: "ok", ...grade.your_result }} maxHeight={220} /></div></details>}
          </div>
        )}
        {solution && (
          <div className="rounded-lg border border-sky-400/20 bg-sky-500/[0.05] p-3">
            <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-sky-300">{isOpen ? "Model answer" : solved ? "Reference solution — compare with yours" : "Reference solution"}</div>
            {isOpen ? <Markdown content={solution.solution} className="callout-body" /> : <div className="[&_pre]:my-1"><CodeBlock code={solution.solution} lang="sql" /></div>}
            {solution.explanation && <div className="mt-2 border-t border-white/5 pt-2"><div className="mb-1 text-xs font-semibold text-slate-400">Why this works</div><Markdown content={solution.explanation} className="callout-body" /></div>}
          </div>
        )}
      </div>
    </div>
  );
}
