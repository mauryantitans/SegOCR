import clsx from "clsx";
import type { LockSnapshot } from "../../lib/types";

const SESSION_COLOR: Record<string, string> = { A: "text-sky-300", B: "text-fuchsia-300", C: "text-amber-300" };

export default function LockTable({ snap }: { snap: LockSnapshot }) {
  return (
    <div className="space-y-2 text-[12px]">
      {snap.transactions.length === 0 && snap.locks.length === 0 && <div className="text-slate-500">No open InnoDB transactions or locks for these sessions.</div>}
      {snap.transactions.length > 0 && (
        <table className="w-full font-mono">
          <thead className="text-left text-slate-500"><tr><th className="pr-3">session</th><th className="pr-3">trx state</th><th className="pr-3">isolation</th><th className="pr-3">rows locked</th><th>rows modified</th></tr></thead>
          <tbody>{snap.transactions.map((t) => (
            <tr key={t.trxId}><td className={clsx("pr-3 font-semibold", SESSION_COLOR[t.session])}>{t.session}</td>
              <td className={clsx("pr-3", t.state === "LOCK WAIT" && "text-amber-300")}>{t.state}</td><td className="pr-3">{t.isolation}</td><td className="pr-3">{t.rowsLocked}</td><td>{t.rowsModified}</td></tr>
          ))}</tbody>
        </table>
      )}
      {snap.locks.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full font-mono">
            <thead className="text-left text-slate-500"><tr><th className="pr-3">session</th><th className="pr-3">table</th><th className="pr-3">index</th><th className="pr-3">type</th><th className="pr-3">mode</th><th className="pr-3">status</th><th>lock data</th></tr></thead>
            <tbody>{snap.locks.map((l, i) => (
              <tr key={i} className={clsx(l.status === "WAITING" && "bg-amber-500/10")}>
                <td className={clsx("pr-3 font-semibold", SESSION_COLOR[l.session])}>{l.session}</td><td className="pr-3">{l.table}</td><td className="pr-3">{l.index ?? "—"}</td>
                <td className="pr-3">{l.lockType}{l.reportedByThreadOf && <span className="ml-1 text-[10px] text-slate-500" title="performance_schema.data_locks.THREAD_ID names this session's thread: it converted the owner's implicit lock">(via {l.reportedByThreadOf}'s thread)</span>}</td><td className="pr-3 text-sky-200">{l.lockMode}</td>
                <td className={clsx("pr-3", l.status === "WAITING" ? "text-amber-300" : "text-emerald-300")}>{l.status}</td><td>{l.lockData ?? "—"}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <div className="text-[10.5px] text-slate-500">From performance_schema.data_locks and information_schema.innodb_trx, filtered to this lab's sessions.</div>
    </div>
  );
}
