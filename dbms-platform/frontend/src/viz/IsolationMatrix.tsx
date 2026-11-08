import clsx from "clsx";
import { useState } from "react";

type V = "yes" | "no" | "partial";
const LEVELS = ["READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] as const;
const ANOMALIES = ["Dirty read", "Non-repeatable read", "Phantom read", "Lost update", "Write skew"] as const;
const M: Record<string, Record<string, [V, string]>> = {
  "READ UNCOMMITTED": {
    "Dirty read": ["yes", "Plain SELECTs may read the latest uncommitted row version."],
    "Non-repeatable read": ["yes", "Each read sees the latest version, committed or not."],
    "Phantom read": ["yes", "New rows inserted by others can appear between reads."],
    "Lost update": ["yes", "Read-modify-write in the application can overwrite a concurrent update."],
    "Write skew": ["yes", "No protection across rows."],
  },
  "READ COMMITTED": {
    "Dirty read": ["no", "Each consistent read uses a fresh snapshot of committed data."],
    "Non-repeatable read": ["yes", "A new snapshot per statement: re-reading a row can return a newer committed value."],
    "Phantom read": ["yes", "New committed rows appear in later statements; InnoDB also uses no gap locks for most searches at this level."],
    "Lost update": ["yes", "Use an atomic UPDATE, SELECT … FOR UPDATE, or a version check."],
    "Write skew": ["yes", "Two transactions read overlapping data and write different rows."],
  },
  "REPEATABLE READ": {
    "Dirty read": ["no", "Snapshot of committed data."],
    "Non-repeatable read": ["no", "InnoDB: all consistent reads in the transaction use the snapshot established by the first read."],
    "Phantom read": ["partial", "SQL standard allows phantoms here. InnoDB: plain SELECTs see the snapshot (no phantoms); locking reads (FOR UPDATE/FOR SHARE, UPDATE, DELETE) read the CURRENT data and take next-key locks that block inserts into the scanned range."],
    "Lost update": ["yes", "InnoDB does not detect lost updates: if you SELECT (snapshot) and later UPDATE with an app-computed value, a concurrent commit is overwritten. Use FOR UPDATE, atomic UPDATE, or optimistic versioning."],
    "Write skew": ["yes", "Not prevented: each transaction's snapshot is consistent but their combined writes violate an invariant."],
  },
  SERIALIZABLE: {
    "Dirty read": ["no", "—"],
    "Non-repeatable read": ["no", "—"],
    "Phantom read": ["no", "InnoDB turns plain SELECTs into SELECT … FOR SHARE (when autocommit is off), so reads take shared next-key locks."],
    "Lost update": ["no", "The shared locks make the second writer wait or deadlock (one transaction is rolled back and must retry)."],
    "Write skew": ["no", "Read locks conflict with the other transaction's writes. Cost: more blocking and deadlocks."],
  },
};
const COLOR: Record<V, string> = { yes: "bg-rose-500/20 text-rose-200", no: "bg-emerald-500/15 text-emerald-200", partial: "bg-amber-500/20 text-amber-200" };
const LABEL: Record<V, string> = { yes: "possible", no: "prevented", partial: "InnoDB-specific" };

export default function IsolationMatrix() {
  const [sel, setSel] = useState<[string, string]>(["REPEATABLE READ", "Phantom read"]);
  const [v, note] = M[sel[0]][sel[1]];
  return (
    <div className="not-prose my-5 overflow-x-auto rounded-xl border border-fuchsia-400/20 bg-ink-900/60 p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-fuchsia-300">Isolation levels × anomalies (InnoDB, MySQL 8.4) — click a cell</div>
      <table className="w-full text-[12.5px]">
        <thead><tr><th className="p-1.5 text-left text-slate-400">Level</th>{ANOMALIES.map((a) => <th key={a} className="p-1.5 text-left font-medium text-slate-300">{a}</th>)}</tr></thead>
        <tbody>{LEVELS.map((l) => (
          <tr key={l}><td className="p-1.5 font-mono text-slate-200">{l}{l === "REPEATABLE READ" && <span className="ml-1 text-[10px] text-sky-300">(default)</span>}</td>
            {ANOMALIES.map((a) => { const [cv] = M[l][a]; return (
              <td key={a} className="p-1"><button onClick={() => setSel([l, a])} className={clsx("w-full rounded-md px-2 py-1 text-left", COLOR[cv], sel[0] === l && sel[1] === a && "ring-2 ring-white/50")}>{LABEL[cv]}</button></td>); })}</tr>))}
        </tbody>
      </table>
      <div className="mt-3 rounded-lg bg-white/[0.03] p-3 text-sm"><span className="font-mono text-fuchsia-200">{sel[0]} · {sel[1]}</span> — <span className={clsx("chip", COLOR[v])}>{LABEL[v]}</span><p className="mt-1 text-slate-300">{note}</p></div>
    </div>
  );
}
