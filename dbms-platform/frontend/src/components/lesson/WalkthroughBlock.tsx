import clsx from "clsx";
import { ChevronLeft, ChevronRight, Footprints } from "lucide-react";
import { useState } from "react";
import type { Capture } from "../../lib/types";
import Markdown, { CodeBlock } from "../Markdown";
import { CapturedOutput } from "./SqlBlock";
import { useLessonCtx } from "./context";

export default function WalkthroughBlock({ title, steps }: { title?: string; steps: { title?: string; explain?: string; sql?: string; capture?: Capture | null }[] }) {
  const [i, setI] = useState(0);
  const ctx = useLessonCtx();
  const st = steps[i];
  return (
    <div className="not-prose my-6 overflow-hidden rounded-xl border border-emerald-400/20 bg-emerald-500/[0.03]">
      <div className="flex items-center gap-2 border-b border-white/5 px-4 py-2.5">
        <Footprints size={15} className="text-emerald-300" />
        <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300">Execution walkthrough</span>
        {title && <span className="truncate font-mono text-xs text-slate-400">{title}</span>}
      </div>
      <div className="flex gap-1 overflow-x-auto px-4 pt-3 scrollbar-thin">
        {steps.map((s, k) => (
          <button key={k} onClick={() => setI(k)} className={clsx("shrink-0 rounded-md px-2.5 py-1 text-xs", k === i ? "bg-emerald-500/20 text-emerald-200" : "text-slate-500 hover:bg-white/5")}>
            {s.title ?? `Step ${k + 1}`}
          </button>
        ))}
      </div>
      <div className="space-y-3 p-4">
        {st.explain && <Markdown content={st.explain} className="callout-body text-sm" />}
        {st.sql && <div className="[&_pre]:my-0"><CodeBlock code={st.sql} lang="sql" /></div>}
        {st.sql && <CapturedOutput capture={st.capture} version={ctx.captureVersion} />}
      </div>
      <div className="flex items-center justify-between border-t border-white/5 px-4 py-2">
        <button className="btn-ghost text-xs" disabled={i === 0} onClick={() => setI(i - 1)}><ChevronLeft size={14} />Previous</button>
        <span className="text-xs text-slate-500">{i + 1} / {steps.length}</span>
        <button className="btn-ghost text-xs" disabled={i === steps.length - 1} onClick={() => setI(i + 1)}>Next<ChevronRight size={14} /></button>
      </div>
    </div>
  );
}
