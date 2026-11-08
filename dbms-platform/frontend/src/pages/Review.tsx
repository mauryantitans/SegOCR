import { useQuery } from "@tanstack/react-query";
import { Bookmark, Flame, NotebookPen } from "lucide-react";
import { Link } from "react-router-dom";
import Markdown from "../components/Markdown";
import { PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { flatLessons, useCourse } from "../lib/queries";
import type { LessonSummary } from "../lib/types";

export default function Review() {
  const { data: course } = useCourse();
  const titles = Object.fromEntries(flatLessons(course).map((l) => [l.id, l.title]));
  const { data: bookmarks } = useQuery({ queryKey: ["bookmarks"], queryFn: () => api<{ id: number; target: string; label: string }[]>("/progress/bookmarks") });
  const { data: notes } = useQuery({ queryKey: ["notes"], queryFn: () => api<{ lessonId: string; body: string; updatedAt: string }[]>("/progress/notes") });
  const { data: weak } = useQuery({ queryKey: ["weak"], queryFn: () => api<{ concept: string; attempts: number; failures: number; recentAccuracy: number; lessons: LessonSummary[] }[]>("/progress/weak-concepts") });
  return (
    <div className="mx-auto max-w-6xl p-4 lg:p-8">
      <PageHeader title="Review & notes" subtitle="Everything you marked for later, your personal notes, and the concepts your quiz and exercise history says you should revise. Each lesson also has a Revision mode with its key rules and common mistakes." />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="glass p-5">
          <h2 className="flex items-center gap-2 font-semibold text-white"><Flame size={16} className="text-amber-300" />Concepts to revise</h2>
          {!weak?.length ? <p className="mt-2 text-sm text-slate-400">Nothing flagged yet.</p> : weak.map((w) => (
            <div key={w.concept} className="mt-3 rounded-lg bg-white/[0.03] p-3 text-sm">
              <div className="flex justify-between"><span className="text-amber-200">{w.concept}</span><span className="text-xs text-slate-500">{Math.round(w.recentAccuracy * 100)}% recent accuracy · {w.failures}/{w.attempts} missed</span></div>
              <div className="mt-1 space-y-0.5">{w.lessons.map((l) => <Link key={l.id} className="block text-xs text-sky-300 hover:underline" to={`/lessons/${l.id}`}>{l.title}</Link>)}</div>
            </div>))}
        </div>
        <div className="glass p-5">
          <h2 className="flex items-center gap-2 font-semibold text-white"><Bookmark size={16} className="text-sky-300" />Bookmarks</h2>
          {!bookmarks?.length ? <p className="mt-2 text-sm text-slate-400">Bookmark lessons with the button at the top of each lesson.</p> : (
            <ul className="mt-3 space-y-1.5">{bookmarks.map((b) => <li key={b.id}><Link to={`/lessons/${b.target}`} className="text-sm text-slate-200 hover:text-sky-300">{b.label}</Link></li>)}</ul>)}
        </div>
        <div className="glass p-5 lg:row-span-2">
          <h2 className="flex items-center gap-2 font-semibold text-white"><NotebookPen size={16} className="text-emerald-300" />My notes</h2>
          {!notes?.length ? <p className="mt-2 text-sm text-slate-400">Open the Notes panel in any lesson to start writing.</p> : notes.map((n) => (
            <div key={n.lessonId} className="mt-3 border-t border-white/5 pt-3">
              <Link to={`/lessons/${n.lessonId}`} className="text-xs font-semibold text-sky-300">{titles[n.lessonId] ?? n.lessonId}</Link>
              <Markdown content={n.body} className="callout-body mt-1 text-sm" />
            </div>))}
        </div>
      </div>
    </div>
  );
}
