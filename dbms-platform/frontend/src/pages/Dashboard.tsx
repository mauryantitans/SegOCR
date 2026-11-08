import { ArrowRight, Award, BookOpen, CheckCircle2, Flame, Target, Terminal } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { LevelBadge, PageHeader, ProgressBar } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { flatLessons, useCourse, useProgress } from "../lib/queries";
import type { LessonSummary } from "../lib/types";

interface Weak { concept: string; attempts: number; failures: number; recentAccuracy: number; lessons: LessonSummary[] }

export default function Dashboard() {
  const { user } = useAuth();
  const { data: course } = useCourse();
  const { data: progress } = useProgress();
  const { data: weak } = useQuery({ queryKey: ["weak"], queryFn: () => api<Weak[]>("/progress/weak-concepts") });
  const lessons = flatLessons(course);
  const authored = lessons.filter((l) => l.status === "authored");
  const status = (id: string) => progress?.lessons[id]?.status;
  const next = authored.find((l) => status(l.id) === "in_progress") ?? authored.find((l) => status(l.id) !== "completed");
  const s = progress?.stats;
  const accuracy = s && s.questionsAnswered ? Math.round((100 * s.questionsCorrect) / s.questionsAnswered) : null;

  return (
    <div className="mx-auto max-w-7xl p-4 lg:p-8">
      <PageHeader title={`Welcome back, ${user?.displayName ?? ""}`} subtitle={course?.course.description} />
      <div className="grid gap-4 md:grid-cols-4">
        {[
          { icon: BookOpen, label: "Lessons completed", value: `${s?.completedLessons ?? 0} / ${s?.authoredLessons ?? 0}`, sub: `${lessons.length} lessons in the full curriculum` },
          { icon: Terminal, label: "Exercises solved", value: s?.solvedExercises ?? 0, sub: "graded against real MySQL" },
          { icon: Target, label: "Quiz accuracy", value: accuracy == null ? "—" : `${accuracy}%`, sub: `${s?.questionsAnswered ?? 0} questions answered` },
          { icon: Award, label: "Assessments passed", value: Object.values(progress?.assessments ?? {}).filter((a) => a.passed).length, sub: `${course?.modules.filter((m) => m.hasAssessment).length ?? 0} available` },
        ].map(({ icon: Icon, label, value, sub }) => (
          <div key={label} className="glass p-4">
            <div className="flex items-center gap-2 text-xs text-slate-400"><Icon size={14} />{label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
            <div className="mt-1 text-[11px] text-slate-500">{sub}</div>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          {next && (
            <Link to={`/lessons/${next.id}`} className="glass group block p-5 transition hover:border-sky-400/30">
              <div className="text-xs uppercase tracking-wider text-sky-300">{status(next.id) === "in_progress" ? "Continue where you left off" : "Up next"}</div>
              <div className="mt-2 flex items-center justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold text-white">{next.title}</div>
                  <div className="mt-1 flex items-center gap-2 text-xs text-slate-400"><LevelBadge level={next.level} /> Module {next.moduleId.slice(1)} · {next.minutes} min</div>
                </div>
                <ArrowRight className="text-sky-300 transition group-hover:translate-x-1" />
              </div>
            </Link>
          )}
          <div className="glass p-5">
            <div className="mb-4 flex items-center justify-between"><h2 className="font-semibold text-white">Modules</h2><Link to="/curriculum" className="text-xs text-sky-300">Full curriculum →</Link></div>
            <div className="grid gap-3 sm:grid-cols-2">
              {course?.modules.map((m) => {
                const ls = m.chapters.flatMap((c) => c.lessons);
                const auth = ls.filter((l) => l.status === "authored");
                const done = auth.filter((l) => status(l.id) === "completed").length;
                return (
                  <Link key={m.id} to={`/curriculum#${m.id}`} className="rounded-xl border border-white/5 bg-white/[0.02] p-3 hover:border-white/15">
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-sm text-slate-100"><span className="text-slate-500">{m.number}.</span> {m.title}</div>
                      <LevelBadge level={m.level} />
                    </div>
                    <ProgressBar className="mt-3" value={auth.length ? (100 * done) / auth.length : 0} />
                    <div className="mt-1.5 text-[11px] text-slate-500">{done}/{auth.length} completed · {auth.length}/{ls.length} lessons available</div>
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
        <div className="space-y-6">
          <div className="glass p-5">
            <h2 className="flex items-center gap-2 font-semibold text-white"><Flame size={16} className="text-amber-300" />Concepts to revise</h2>
            {!weak?.length ? (
              <p className="mt-3 text-sm text-slate-400">No weak spots detected yet. Concepts you repeatedly miss in quizzes and exercises will appear here with links to the lessons that teach them.</p>
            ) : (
              <ul className="mt-3 space-y-3">
                {weak.slice(0, 6).map((w) => (
                  <li key={w.concept} className="rounded-lg bg-white/[0.03] p-3">
                    <div className="flex justify-between text-sm"><span className="font-medium text-amber-200">{w.concept}</span><span className="text-xs text-slate-500">{w.failures} misses</span></div>
                    <div className="mt-1 flex flex-wrap gap-1">{w.lessons.map((l) => <Link key={l.id} to={`/lessons/${l.id}`} className="text-[11px] text-sky-300 hover:underline">{l.title}</Link>)}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="glass p-5">
            <h2 className="flex items-center gap-2 font-semibold text-white"><CheckCircle2 size={16} className="text-emerald-300" />How each lesson works</h2>
            <ol className="mt-3 space-y-1.5 text-sm text-slate-400">
              {["Concept & DBMS theory", "MySQL implementation", "Internal working (InnoDB / server)", "Execution walkthrough & visualization", "Optimization & best practices", "Industry application", "Exercises & quiz graded on real MySQL"].map((t, i) => (
                <li key={t} className="flex gap-2"><span className="text-slate-600">{i + 1}.</span>{t}</li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
