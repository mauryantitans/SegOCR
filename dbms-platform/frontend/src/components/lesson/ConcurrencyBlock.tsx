import clsx from "clsx";
import { Eye, Loader2, Play, RotateCcw, SkipForward, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import type { Capture, ConcStep, ConcStepCapture, LockSnapshot, RunResult, StatementResult } from "../../lib/types";
import Markdown, { CodeBlock } from "../Markdown";
import { StatementList } from "../ResultTable";
import { useLessonCtx } from "./context";
import LockTable from "./LockTable";

const COLORS: Record<string, string> = { A: "border-sky-400/40 bg-sky-500/[0.04]", B: "border-fuchsia-400/40 bg-fuchsia-500/[0.04]", C: "border-amber-400/40 bg-amber-500/[0.04]" };
const LABEL: Record<string, string> = { A: "text-sky-300", B: "text-fuchsia-300", C: "text-amber-300" };

interface Live { status: string; statements?: StatementResult[]; snapshot?: LockSnapshot; completedAfterStep?: number }

function StepResult({ r }: { r: Live | ConcStepCapture }) {
  if (r.snapshot) return <LockTable snap={r.snapshot} />;
  const sts = (r.statements ?? []).filter((s) => s.status !== "ok" || s.columns?.length || s.affected_rows != null);
  return (
    <div className="space-y-1">
      {r.completedAfterStep != null && <div className="text-[11px] text-amber-300">Was blocked — completed after step {r.completedAfterStep + 1}</div>}
      {sts.length ? <StatementList statements={sts.map((s, i) => ({ ...s, index: i, columns: s.columns ?? [], rows: s.rows ?? [] }))} showSql={sts.length > 1} />
        : r.status === "blocked" ? <StatementList statements={[{ index: 0, sql: "", status: "blocked", columns: [], rows: [] }]} />
        : <div className="text-[11.5px] text-emerald-400/80">OK</div>}
    </div>
  );
}

export default function ConcurrencyBlock({ id, title, description, steps, capture }: { id: string; title?: string; description?: string; steps: ConcStep[]; capture?: Capture | null }) {
  const ctx = useLessonCtx();
  const lab = `conc:${ctx.docId}:${id}`;
  const sessions = Array.from(new Set(steps.filter((s) => s.session).map((s) => s.session!))).sort();
  const [mode, setMode] = useState<"captured" | "live">("captured");
  const [live, setLive] = useState<Record<number, Live>>({});
  const [cursor, setCursor] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const pending = useRef<Record<string, number>>({});
  const captured: Record<number, ConcStepCapture> = Object.fromEntries((capture?.steps ?? []).map((s) => [s.step, s]));

  useEffect(() => {
    if (mode !== "live") return;
    const t = setInterval(async () => {
      for (const [sess, idx] of Object.entries(pending.current)) {
        const p = await api<{ finished: StatementResult[]; busy: boolean }>(`/labs/poll?lab=${encodeURIComponent(lab)}&session=${sess}`).catch(() => null);
        if (p?.finished.length) {
          delete pending.current[sess];
          setLive((l) => ({ ...l, [idx]: { ...l[idx], status: p.finished[p.finished.length - 1].status, statements: [...(l[idx]?.statements ?? []).filter((s) => s.status !== "blocked"), ...p.finished], completedAfterStep: Math.max(0, cursorRef.current - 1) } }));
        }
      }
    }, 700);
    return () => clearInterval(t);
  }, [mode, lab]);
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;

  const start = async () => {
    setErr(null); setBusy(true);
    try { await api("/labs/reset", { method: "POST", json: { lab } }); setLive({}); setCursor(0); pending.current = {}; setMode("live"); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const next = async () => {
    const st = steps[cursor];
    setErr(null); setBusy(true);
    try {
      if (st.observe) {
        const snap = await api<LockSnapshot>(`/labs/locks?lab=${encodeURIComponent(lab)}`);
        setLive((l) => ({ ...l, [cursor]: { status: "ok", snapshot: snap } }));
      } else {
        const r = await api<RunResult>("/labs/run", { method: "POST", json: { lab, session: st.session, sql: st.sql, stop_on_error: true } });
        const blocked = r.statements.some((s) => s.status === "blocked");
        if (blocked) pending.current[st.session!] = cursor;
        setLive((l) => ({ ...l, [cursor]: { status: blocked ? "blocked" : r.statements.some((s) => s.status === "error") ? "error" : "ok", statements: r.statements } }));
      }
      setCursor(cursor + 1);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <div className="not-prose my-6 overflow-hidden rounded-xl border border-white/10 bg-ink-900/50">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 px-4 py-2.5">
        <div className="flex items-center gap-2"><Users size={15} className="text-fuchsia-300" />
          <span className="text-xs font-semibold uppercase tracking-wider text-fuchsia-300">Concurrent sessions</span>
          {title && <span className="text-sm text-slate-200">{title}</span>}</div>
        <div className="flex gap-1">
          {mode === "live" && <button className="btn-ghost text-xs" onClick={() => setMode("captured")}><RotateCcw size={13} />Captured run</button>}
          {mode === "captured" ? <button className="btn-primary text-xs" onClick={start} disabled={busy}>{busy ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}Run it live</button>
            : <>
              <button className="btn-outline text-xs" onClick={start} disabled={busy}><RotateCcw size={13} />Restart</button>
              <button className="btn-primary text-xs" onClick={next} disabled={busy || cursor >= steps.length}>{busy ? <Loader2 size={13} className="animate-spin" /> : <SkipForward size={13} />}{cursor >= steps.length ? "Finished" : `Run step ${cursor + 1}`}</button>
            </>}
        </div>
      </div>
      {description && <div className="px-4 pt-3 text-sm text-slate-400">{description}</div>}
      {err && <div className="mx-4 mt-3 rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{err}</div>}
      <div className="hidden px-4 pt-3 md:grid" style={{ gridTemplateColumns: `2.5rem repeat(${sessions.length}, minmax(0, 1fr))` }}>
        <div />{sessions.map((s) => <div key={s} className={clsx("px-2 text-xs font-semibold", LABEL[s])}>Session {s}</div>)}
      </div>
      <div className="space-y-2 p-4">
        {steps.map((st, k) => {
          const res = mode === "live" ? live[k] : captured[k];
          const dim = mode === "live" && k >= cursor;
          if (st.observe) {
            return (
              <div key={k} className={clsx("grid gap-2 md:grid-cols-[2.5rem_1fr]", dim && "opacity-40")}>
                <div className="pt-2 text-xs text-slate-500">{k + 1}</div>
                <div className="rounded-lg border border-emerald-400/30 bg-emerald-500/[0.04] p-3">
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-emerald-300"><Eye size={13} />Inspect InnoDB locks and transactions</div>
                  {st.note && <div className="mb-2 text-sm text-slate-300"><Markdown content={st.note} className="callout-body" /></div>}
                  {res && <StepResult r={res} />}
                </div>
              </div>
            );
          }
          const col = sessions.indexOf(st.session!);
          return (
            <div key={k} className={clsx("grid gap-2", dim && "opacity-40")} style={{ gridTemplateColumns: `2.5rem repeat(${sessions.length}, minmax(0, 1fr))` }}>
              <div className={clsx("pt-2 text-xs", mode === "live" && k === cursor ? "font-bold text-sky-300" : "text-slate-500")}>{k + 1}</div>
              {sessions.map((s, c) => c === col ? (
                <div key={s} className={clsx("min-w-0 rounded-lg border-l-2 p-2.5", COLORS[s])}>
                  <div className={clsx("mb-1 text-[11px] font-semibold md:hidden", LABEL[s])}>Session {s}</div>
                  <div className="[&_pre]:my-0 [&_pre]:border-0 [&_pre]:bg-transparent [&_pre]:p-0 [&_pre]:text-[12px]"><CodeBlock code={st.sql!} lang="sql" /></div>
                  {st.note && <div className="mt-1.5 text-[12.5px] text-slate-400"><Markdown content={st.note} className="callout-body" /></div>}
                  {res && <div className="mt-2 border-t border-white/5 pt-2"><StepResult r={res} /></div>}
                </div>
              ) : <div key={s} className="hidden md:block" />)}
            </div>
          );
        })}
      </div>
      <div className="border-t border-white/5 px-4 py-2 text-[11px] text-slate-500">
        {mode === "captured" ? "Results above were captured by running these sessions concurrently on a real MySQL server. Press “Run it live” to drive the sessions yourself, one step at a time." : "Live: each session is a separate MySQL connection in your private lab."}
      </div>
    </div>
  );
}
