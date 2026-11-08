import clsx from "clsx";
import { Eye, Loader2, Play, RotateCcw } from "lucide-react";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import LockTable from "../components/lesson/LockTable";
import { StatementList } from "../components/ResultTable";
import { PageHeader } from "../components/ui";
import { api } from "../lib/api";
import type { LockSnapshot, RunResult, StatementResult } from "../lib/types";
import IsolationMatrix from "../viz/IsolationMatrix";

const SqlEditor = lazy(() => import("../components/SqlEditor"));
type S = "A" | "B";
const SETUP = `CREATE TABLE IF NOT EXISTS accounts (id INT PRIMARY KEY, owner VARCHAR(20), balance DECIMAL(10,2));
INSERT IGNORE INTO accounts VALUES (10, 'ana', 500.00), (20, 'bo', 300.00), (30, 'chen', 900.00);`;
const SNIPPETS: Record<string, string> = {
  "Start transaction": "START TRANSACTION;",
  "Isolation: READ COMMITTED": "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;",
  "Isolation: REPEATABLE READ": "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;",
  "Isolation: SERIALIZABLE": "SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;",
  "Read": "SELECT * FROM accounts WHERE id = 10;",
  "Lock row": "SELECT * FROM accounts WHERE id = 10 FOR UPDATE;",
  "Range lock": "SELECT * FROM accounts WHERE id BETWEEN 10 AND 25 FOR UPDATE;",
  "Update": "UPDATE accounts SET balance = balance - 100 WHERE id = 10;",
  "Insert into gap": "INSERT INTO accounts VALUES (15, 'dee', 50.00);",
  "Commit": "COMMIT;", "Rollback": "ROLLBACK;",
};
interface LogEntry { session: S; sql: string; result: StatementResult[]; at: number }

export default function ConcurrencyLab() {
  const [code, setCode] = useState<Record<S, string>>({ A: "START TRANSACTION;\nSELECT * FROM accounts WHERE id = 10 FOR UPDATE;", B: "UPDATE accounts SET balance = balance + 50 WHERE id = 10;" });
  const [busy, setBusy] = useState<Record<S, boolean>>({ A: false, B: false });
  const [waiting, setWaiting] = useState<Record<S, boolean>>({ A: false, B: false });
  const [log, setLog] = useState<LogEntry[]>([]);
  const [locks, setLocks] = useState<LockSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const waitingRef = useRef(waiting);
  waitingRef.current = waiting;

  const reset = async () => {
    setErr(null); setLog([]); setLocks(null); setWaiting({ A: false, B: false });
    try {
      await api("/labs/reset", { method: "POST", json: { lab: "concurrency", dataset: "empty" } });
      await api("/labs/run", { method: "POST", json: { lab: "concurrency", session: "A", sql: SETUP } });
      setReady(true);
    } catch (e) { setErr((e as Error).message); }
  };
  useEffect(() => { reset(); }, []);
  useEffect(() => {
    const t = setInterval(async () => {
      for (const s of ["A", "B"] as S[]) {
        if (!waitingRef.current[s]) continue;
        const p = await api<{ finished: StatementResult[] }>(`/labs/poll?lab=concurrency&session=${s}`).catch(() => null);
        if (p?.finished.length) {
          setWaiting((w) => ({ ...w, [s]: false }));
          setLog((l) => [...l, { session: s, sql: "(blocked statement completed)", result: p.finished, at: Date.now() }]);
        }
      }
    }, 600);
    return () => clearInterval(t);
  }, []);

  const run = async (s: S, sql = code[s]) => {
    setErr(null); setBusy((b) => ({ ...b, [s]: true }));
    try {
      const r = await api<RunResult>("/labs/run", { method: "POST", json: { lab: "concurrency", session: s, sql, stop_on_error: true } });
      if (r.statements.some((x) => x.status === "blocked")) setWaiting((w) => ({ ...w, [s]: true }));
      setLog((l) => [...l, { session: s, sql, result: r.statements, at: Date.now() }]);
    } catch (e) { setErr((e as Error).message); } finally { setBusy((b) => ({ ...b, [s]: false })); }
  };

  return (
    <div className="mx-auto max-w-7xl p-4 lg:p-8">
      <PageHeader title="Concurrency Lab" subtitle="Two real MySQL connections (sessions A and B) on the same private schema. Start transactions, take locks, block each other, cause a deadlock — and inspect InnoDB's actual lock table while it happens.">
        <div className="flex gap-2">
          <button className="btn-outline text-xs" onClick={async () => setLocks(await api<LockSnapshot>("/labs/locks?lab=concurrency"))}><Eye size={13} />Inspect locks</button>
          <button className="btn-ghost text-xs" onClick={reset}><RotateCcw size={13} />Reset (closes both sessions)</button>
        </div>
      </PageHeader>
      {err && <div className="mb-3 rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{err}</div>}
      <div className="mb-4 text-xs text-slate-500">Table: <code className="text-slate-300">accounts(id INT PRIMARY KEY, owner, balance)</code> with ids 10, 20, 30 — gaps between them make gap and next-key locks visible.</div>
      <div className="grid gap-4 lg:grid-cols-2">
        {(["A", "B"] as S[]).map((s) => (
          <div key={s} className={clsx("glass p-3", s === "A" ? "border-sky-400/30" : "border-fuchsia-400/30")}>
            <div className="mb-2 flex items-center justify-between">
              <span className={clsx("text-sm font-semibold", s === "A" ? "text-sky-300" : "text-fuchsia-300")}>Session {s}</span>
              {waiting[s] && <span className="chip animate-pulse bg-amber-500/20 text-amber-200">waiting for lock…</span>}
              <button className="btn-primary text-xs" onClick={() => run(s)} disabled={busy[s] || waiting[s] || !ready}>{busy[s] ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}Run</button>
            </div>
            <Suspense fallback={<div className="h-32" />}><SqlEditor value={code[s]} onChange={(v) => setCode((c) => ({ ...c, [s]: v }))} onRun={(t) => run(s, t)} height={150} /></Suspense>
            <div className="mt-2 flex flex-wrap gap-1">
              {Object.entries(SNIPPETS).map(([k, v]) => <button key={k} className="rounded bg-white/5 px-1.5 py-0.5 text-[10.5px] text-slate-400 hover:bg-white/10 hover:text-slate-200" onClick={() => setCode((c) => ({ ...c, [s]: v }))}>{k}</button>)}
            </div>
          </div>
        ))}
      </div>
      {locks && <div className="glass mt-4 p-4"><div className="mb-2 text-xs font-semibold uppercase tracking-wider text-emerald-300">InnoDB locks right now</div><LockTable snap={locks} /></div>}
      <div className="glass mt-4 p-4">
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Timeline</div>
        {!log.length && <div className="text-sm text-slate-500">Run statements in A and B to build a timeline.</div>}
        <div className="space-y-2">
          {log.map((e, i) => (
            <div key={i} className={clsx("grid gap-2 md:grid-cols-2")}>
              <div className={clsx(e.session === "B" && "md:col-start-2", "rounded-lg border-l-2 p-2", e.session === "A" ? "border-sky-400 bg-sky-500/[0.04]" : "border-fuchsia-400 bg-fuchsia-500/[0.04]")}>
                <div className="mb-1 flex justify-between text-[11px] text-slate-500"><span>{i + 1}. session {e.session}</span><span>{new Date(e.at).toLocaleTimeString()}</span></div>
                <pre className="mb-1 whitespace-pre-wrap font-mono text-[11.5px] text-slate-300">{e.sql}</pre>
                <StatementList statements={e.result.filter((x) => x.status !== "ok" || x.columns.length || x.affected_rows)} />
              </div>
            </div>
          ))}
        </div>
      </div>
      <IsolationMatrix />
    </div>
  );
}
