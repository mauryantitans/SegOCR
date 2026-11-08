import clsx from "clsx";
import type { Level, Stage } from "../lib/types";

const LEVEL: Record<Level, string> = {
  beginner: "bg-emerald-500/15 text-emerald-300", intermediate: "bg-sky-500/15 text-sky-300",
  advanced: "bg-amber-500/15 text-amber-300", professional: "bg-violet-500/15 text-violet-300",
};
export function LevelBadge({ level }: { level: Level }) {
  return <span className={clsx("chip capitalize", LEVEL[level])}>{level}</span>;
}

export const STAGE_META: Record<Stage, { label: string; color: string; text: string }> = {
  concept: { label: "Concept & Theory", color: "bg-stage-concept", text: "text-stage-concept" },
  internals: { label: "Internal Working", color: "bg-stage-internals", text: "text-stage-internals" },
  implementation: { label: "MySQL Implementation", color: "bg-stage-implementation", text: "text-stage-implementation" },
  execution: { label: "Execution & Visualization", color: "bg-stage-execution", text: "text-stage-execution" },
  optimization: { label: "Optimization & Practice", color: "bg-stage-optimization", text: "text-stage-optimization" },
  industry: { label: "Industry Application", color: "bg-stage-industry", text: "text-stage-industry" },
  assessment: { label: "Assessment", color: "bg-stage-assessment", text: "text-stage-assessment" },
};

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx("h-1.5 overflow-hidden rounded-full bg-white/5", className)}>
      <div className="h-full rounded-full bg-gradient-to-r from-sky-400 to-indigo-400 transition-all" style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
    </div>
  );
}

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: string; children?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div><h1 className="text-2xl font-semibold text-white">{title}</h1>{subtitle && <p className="mt-1 max-w-3xl text-sm text-slate-400">{subtitle}</p>}</div>
      {children}
    </div>
  );
}
