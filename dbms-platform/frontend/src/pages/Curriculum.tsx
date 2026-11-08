import clsx from "clsx";
import { CheckCircle2, Circle, CircleDot, Clock, Lock } from "lucide-react";
import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { LevelBadge, PageHeader, ProgressBar } from "../components/ui";
import { flatLessons, useCourse, useProgress } from "../lib/queries";

export default function Curriculum() {
  const { data: course } = useCourse();
  const { data: progress } = useProgress();
  const loc = useLocation();
  useEffect(() => {
    if (loc.hash) setTimeout(() => document.getElementById(loc.hash.slice(1))?.scrollIntoView({ behavior: "smooth" }), 50);
  }, [loc.hash, course]);
  if (!course) return null;
  const all = flatLessons(course);
  const status = (id: string) => progress?.lessons[id]?.status;
  const authoredCount = all.filter((l) => l.status === "authored").length;

  return (
    <div className="mx-auto max-w-6xl p-4 lg:p-8">
      <PageHeader title={course.course.title}
        subtitle={`${course.modules.length} modules · ${all.length} lessons (${authoredCount} available now, the rest are mapped with their full outline) · ${course.projects.length} projects · target: MySQL ${course.course.mysqlVersion}. Prerequisites are recommended, not enforced.`} />
      <div className="space-y-6">
        {course.modules.map((m) => {
          const ls = m.chapters.flatMap((c) => c.lessons);
          const done = ls.filter((l) => status(l.id) === "completed").length;
          return (
            <section key={m.id} id={m.id} className="glass scroll-mt-20 p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="max-w-3xl">
                  <div className="text-xs uppercase tracking-wider text-slate-500">Module {m.number}</div>
                  <h2 className="mt-1 text-lg font-semibold text-white">{m.title}</h2>
                  <p className="mt-1 text-sm text-slate-400">{m.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  <LevelBadge level={m.level} />
                  {m.hasAssessment && <Link to={`/assessments/module/${m.id}-assessment`} className="btn-outline text-xs">Module assessment</Link>}
                </div>
              </div>
              <ProgressBar className="mt-3" value={(100 * done) / ls.length} />
              <details className="mt-3 text-sm text-slate-400"><summary className="cursor-pointer text-xs text-slate-500">Learning outcomes</summary>
                <ul className="mt-2 list-disc space-y-0.5 pl-5">{m.outcomes.map((o) => <li key={o}>{o}</li>)}</ul></details>
              <div className="mt-4 space-y-4">
                {m.chapters.map((c) => (
                  <div key={c.id}>
                    <div className="mb-1.5 text-xs font-medium text-slate-400">{c.title}</div>
                    <div className="divide-y divide-white/5 rounded-xl border border-white/5">
                      {c.lessons.map((l) => {
                        const st = status(l.id);
                        const missing = l.prerequisites.filter((p) => status(p) !== "completed");
                        return (
                          <Link key={l.id} to={`/lessons/${l.id}`} className={clsx("flex items-center gap-3 px-3 py-2.5 hover:bg-white/[0.03]", l.status === "planned" && "opacity-60")}>
                            {st === "completed" ? <CheckCircle2 size={17} className="shrink-0 text-emerald-400" /> : st === "in_progress" ? <CircleDot size={17} className="shrink-0 text-sky-400" /> : <Circle size={17} className="shrink-0 text-slate-600" />}
                            <div className="min-w-0 flex-1">
                              <div className="truncate text-sm text-slate-100">{l.title}</div>
                              <div className="truncate text-[11px] text-slate-500">{l.topics.slice(0, 5).join(" · ")}</div>
                            </div>
                            {l.status === "planned" && <span className="chip bg-white/5 text-slate-400">outline only</span>}
                            {l.status === "authored" && missing.length > 0 && st !== "completed" && <span title={`Recommended first: ${missing.join(", ")}`} className="chip bg-amber-500/10 text-amber-300"><Lock size={10} />{missing.length} prereq</span>}
                            <LevelBadge level={l.level} />
                            <span className="hidden items-center gap-1 text-[11px] text-slate-500 sm:flex"><Clock size={11} />{l.minutes}m</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
