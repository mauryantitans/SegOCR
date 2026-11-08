import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowLeft, ArrowRight, Bookmark, BookmarkCheck, CheckCircle2, Circle, Clock, Database, ExternalLink, GraduationCap, ListChecks, NotebookPen, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Loading } from "../App";
import { LessonContext } from "../components/lesson/context";
import Segments from "../components/lesson/Segments";
import Markdown from "../components/Markdown";
import { LevelBadge, STAGE_META } from "../components/ui";
import { api } from "../lib/api";
import { useLesson, useProgress, useSetLessonStatus } from "../lib/queries";
import type { LessonDetail, Segment } from "../lib/types";

function NotesDrawer({ lessonId, onClose }: { lessonId: string; onClose: () => void }) {
  const { data } = useQuery({ queryKey: ["note", lessonId], queryFn: () => api<{ body: string }>(`/progress/notes/${lessonId}`) });
  const [body, setBody] = useState<string | null>(null);
  const [saved, setSaved] = useState(true);
  useEffect(() => { if (data && body === null) setBody(data.body); }, [data, body]);
  useEffect(() => {
    if (body === null || saved) return;
    const t = setTimeout(() => api(`/progress/notes/${lessonId}`, { method: "PUT", json: { body } }).then(() => setSaved(true)), 700);
    return () => clearTimeout(t);
  }, [body, saved, lessonId]);
  return (
    <div className="glass-strong fixed bottom-4 right-4 top-20 z-30 flex w-[min(420px,calc(100vw-2rem))] flex-col p-4 shadow-2xl">
      <div className="mb-2 flex items-center justify-between"><span className="flex items-center gap-2 text-sm font-semibold"><NotebookPen size={15} />My notes</span>
        <span className="text-[11px] text-slate-500">{saved ? "saved" : "saving…"}</span><button className="btn-ghost px-1.5" onClick={onClose}><X size={15} /></button></div>
      <textarea className="input flex-1 resize-none font-mono text-[13px]" placeholder="Write notes in Markdown — they are saved automatically and collected on the Review page." value={body ?? ""} onChange={(e) => { setBody(e.target.value); setSaved(false); }} />
    </div>
  );
}

const CALLOUT_RE = /^:::(theory|best-practice|warning|internals|performance|standard|version)(\[[^\]]*\])?\n([\s\S]*?)\n:::/gm;
function RevisionSheet({ lesson }: { lesson: LessonDetail }) {
  const callouts: string[] = [];
  let section = "";
  const mistakes: string[] = [];
  lesson.segments.forEach((s: Segment) => {
    if (s.type === "heading") section = s.title;
    if (s.type === "markdown") {
      for (const m of s.content.matchAll(CALLOUT_RE)) callouts.push(m[0]);
      if (section === "Common Mistakes") mistakes.push(s.content.replace(CALLOUT_RE, ""));
    }
  });
  return (
    <div className="space-y-4">
      <div className="callout callout-note"><div className="callout-title">Revision sheet</div><div className="callout-body"><p>{lesson.meta.summary}</p><p>Topics: {lesson.topics.join(" · ")}</p></div></div>
      {callouts.map((c, i) => <Markdown key={i} content={c} />)}
      {mistakes.length > 0 && <><h3 className="text-lg font-semibold text-white">Common mistakes</h3><Markdown content={mistakes.join("\n\n")} /></>}
      <p className="text-sm text-slate-500">Switch back to the full lesson to run examples, do the exercises and take the quiz.</p>
    </div>
  );
}

export default function Lesson() {
  const { id = "" } = useParams();
  const { data: lesson, isLoading, error } = useLesson(id);
  const { data: progress } = useProgress();
  const setStatus = useSetLessonStatus();
  const qc = useQueryClient();
  const { data: bookmarks } = useQuery({ queryKey: ["bookmarks"], queryFn: () => api<{ id: number; target: string }[]>("/progress/bookmarks") });
  const bm = bookmarks?.find((b) => b.target === id);
  const toggleBm = useMutation({
    mutationFn: () => (bm ? api(`/progress/bookmarks/${bm.id}`, { method: "DELETE" }) : api("/progress/bookmarks", { method: "POST", json: { target: id, label: lesson?.title ?? id } })),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["bookmarks"] }),
  });
  const [notes, setNotes] = useState(false);
  const [revision, setRevision] = useState(false);
  const [active, setActive] = useState<string>("");
  const contentRef = useRef<HTMLDivElement>(null);
  const status = progress?.lessons[id]?.status;

  useEffect(() => { window.scrollTo(0, 0); document.querySelector("main")?.scrollTo(0, 0); setRevision(false); }, [id]);
  useEffect(() => {
    if (lesson?.status === "authored" && progress && !status) setStatus.mutate({ id, status: "in_progress" });
  }, [lesson?.status, id, !!progress, status]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const els = contentRef.current?.querySelectorAll("[data-section]");
    if (!els?.length) return;
    const obs = new IntersectionObserver((entries) => entries.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: "-20% 0px -70% 0px" });
    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, [lesson, revision]);

  const ctx = useMemo(() => {
    const runnable: { id: string; sql: string }[] = [];
    lesson?.segments.forEach((s) => {
      if (s.type === "sql" && s.run && s.expect === "ok") runnable.push({ id: s.id, sql: s.sql });
      if (s.type === "walkthrough") s.steps.forEach((st, k) => st.sql && runnable.push({ id: `${s.id}.${k}`, sql: st.sql }));
    });
    return { docId: id, dataset: lesson?.meta.dataset ?? "empty", runnable, captureVersion: lesson?.captureMeta?.mysqlVersion };
  }, [lesson, id]);

  if (isLoading) return <Loading />;
  if (error || !lesson) return <div className="p-10 text-rose-300">Could not load lesson: {String((error as Error)?.message)}</div>;
  const missingPrereqs = lesson.prerequisiteLessons.filter((p) => progress && progress.lessons[p.id]?.status !== "completed");

  return (
    <LessonContext.Provider value={ctx}>
      <div className="mx-auto flex max-w-[1400px] gap-8 px-4 py-6 lg:px-8">
        <aside className="sticky top-20 hidden h-[calc(100vh-6rem)] w-60 shrink-0 overflow-y-auto pr-2 scrollbar-thin xl:block">
          <Link to={`/curriculum#${lesson.module.id}`} className="text-xs text-slate-500 hover:text-slate-300">← Module {lesson.module.number}: {lesson.module.title}</Link>
          <div className="mt-4 space-y-0.5">
            {lesson.sections.map((s) => (
              <a key={s.anchor} href={`#${s.anchor}`} className={clsx("flex items-center gap-2 rounded-md px-2 py-1 text-[13px] transition", active === s.anchor ? "bg-white/5 text-white" : "text-slate-400 hover:text-slate-200")}
                onClick={(e) => { e.preventDefault(); document.getElementById(s.anchor)?.scrollIntoView({ behavior: "smooth" }); }}>
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${STAGE_META[s.stage].color}`} />{s.title}
              </a>
            ))}
          </div>
          {lesson.sections.length > 0 && <div className="mt-6 space-y-1 border-t border-white/5 pt-4 text-[10.5px] text-slate-500">
            {Object.entries(STAGE_META).map(([k, v]) => <div key={k} className="flex items-center gap-2"><span className={`h-1.5 w-1.5 rounded-full ${v.color}`} />{v.label}</div>)}
          </div>}
        </aside>

        <article ref={contentRef} className="min-w-0 max-w-4xl flex-1">
          <div className="mb-6">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
              <span>Module {lesson.module.number}</span><span>·</span><LevelBadge level={lesson.level} />
              <span className="inline-flex items-center gap-1"><Clock size={12} />{lesson.minutes} min</span>
              {lesson.meta.dataset && <span className="chip bg-white/5 text-slate-300"><Database size={11} />{lesson.meta.dataset} dataset</span>}
            </div>
            <h1 className="mt-2 text-3xl font-semibold leading-tight text-white">{lesson.title}</h1>
            {lesson.meta.summary && <p className="mt-2 text-slate-400">{lesson.meta.summary}</p>}
            <div className="mt-4 flex flex-wrap gap-2">
              {lesson.status === "authored" && (
                <button className={clsx("btn text-xs", status === "completed" ? "bg-emerald-500/15 text-emerald-300" : "btn-outline")}
                  onClick={() => status === "completed" ? api(`/progress/lessons/${id}`, { method: "DELETE" }).then(() => qc.invalidateQueries({ queryKey: ["progress"] })) : setStatus.mutate({ id, status: "completed" })}>
                  {status === "completed" ? <CheckCircle2 size={14} /> : <Circle size={14} />}{status === "completed" ? "Completed" : "Mark as complete"}</button>)}
              <button className="btn-outline text-xs" onClick={() => toggleBm.mutate()}>{bm ? <BookmarkCheck size={14} className="text-amber-300" /> : <Bookmark size={14} />}{bm ? "Bookmarked" : "Bookmark"}</button>
              <button className="btn-outline text-xs" onClick={() => setNotes(!notes)}><NotebookPen size={14} />Notes</button>
              {lesson.status === "authored" && <button className={clsx("btn text-xs", revision ? "bg-sky-500/15 text-sky-300" : "btn-outline")} onClick={() => setRevision(!revision)}><ListChecks size={14} />{revision ? "Full lesson" : "Revision mode"}</button>}
            </div>
            {missingPrereqs.length > 0 && (
              <div className="mt-4 rounded-lg border border-amber-400/20 bg-amber-500/[0.06] px-3 py-2 text-sm text-amber-100">
                <GraduationCap size={14} className="mr-1.5 inline" />Recommended first: {missingPrereqs.map((p, i) => <span key={p.id}>{i > 0 && ", "}<Link className="text-amber-300 underline" to={`/lessons/${p.id}`}>{p.title}</Link></span>)}
              </div>
            )}
          </div>

          {lesson.status === "planned" ? (
            <div className="glass p-6">
              <h2 className="text-lg font-semibold text-white">Lesson outline</h2>
              <p className="mt-1 text-sm text-slate-400">This lesson is mapped in the curriculum and is being written. It will follow the same 16-part structure (theory → MySQL implementation → internals → execution → optimization → industry → assessment) and cover:</p>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-300">{lesson.topics.map((t) => <li key={t}>{t}</li>)}</ul>
              <p className="mt-4 text-sm text-slate-400">Meanwhile you can explore these topics in the <Link to="/playground" className="text-sky-300">SQL Playground</Link>.</p>
            </div>
          ) : revision ? <RevisionSheet lesson={lesson} /> : <Segments segments={lesson.segments} />}

          {lesson.meta.references && lesson.meta.references.length > 0 && (
            <div className="mt-12 border-t border-white/5 pt-6">
              <h3 className="text-sm font-semibold text-slate-300">References</h3>
              <ul className="mt-2 space-y-1 text-sm">{lesson.meta.references.map((r) => <li key={r.url}><a href={r.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sky-300 hover:underline">{r.title}<ExternalLink size={11} /></a></li>)}</ul>
              {lesson.captureMeta && <p className="mt-3 text-xs text-slate-500">Example outputs captured on MySQL {lesson.captureMeta.mysqlVersion} ({lesson.captureMeta.capturedAt}). Course target: MySQL 8.4 LTS.</p>}
            </div>
          )}
          <div className="mt-10 flex justify-between gap-4 border-t border-white/5 pt-6">
            {lesson.prev ? <Link to={`/lessons/${lesson.prev.id}`} className="btn-ghost max-w-[45%] text-left text-xs"><ArrowLeft size={14} className="shrink-0" /><span className="truncate">{lesson.prev.title}</span></Link> : <span />}
            {lesson.next && <Link to={`/lessons/${lesson.next.id}`} className="btn-ghost max-w-[45%] text-right text-xs"><span className="truncate">{lesson.next.title}</span><ArrowRight size={14} className="shrink-0" /></Link>}
          </div>
        </article>
      </div>
      {notes && <NotesDrawer lessonId={id} onClose={() => setNotes(false)} />}
    </LessonContext.Provider>
  );
}
