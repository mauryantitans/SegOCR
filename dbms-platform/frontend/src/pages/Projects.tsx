import { useQuery } from "@tanstack/react-query";
import { Boxes, CheckCircle2 } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { Loading } from "../App";
import { LessonContext } from "../components/lesson/context";
import Segments from "../components/lesson/Segments";
import { LevelBadge, PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { flatLessons, useCourse, useProgress } from "../lib/queries";
import type { LessonDetail } from "../lib/types";

const STEPS = ["Requirements gathering", "Entities & relationships", "ER diagram", "Schema design", "Normalization", "MySQL table creation", "Constraints & relationships",
  "SQL implementation", "Indexing strategy", "Transaction management", "Query optimization", "Security", "Testing & validation", "Production considerations"];

function ProjectDetail({ id }: { id: string }) {
  const { data, isLoading } = useQuery({ queryKey: ["project", id], queryFn: () => api<LessonDetail & { authored: boolean; practices: string[] }>(`/projects/${id}`) });
  if (isLoading || !data) return <Loading />;
  const runnable: { id: string; sql: string }[] = [];
  data.segments?.forEach((s) => { if (s.type === "sql" && s.run && s.expect === "ok") runnable.push({ id: s.id, sql: s.sql }); });
  return (
    <LessonContext.Provider value={{ docId: id, dataset: data.meta?.dataset ?? "empty", runnable, captureVersion: data.captureMeta?.mysqlVersion }}>
      <div className="mx-auto max-w-4xl p-4 lg:p-8">
        <Link to="/projects" className="text-xs text-slate-500">← All projects</Link>
        <h1 className="mt-2 text-3xl font-semibold text-white">{data.title}</h1>
        <div className="mt-2 flex flex-wrap gap-1.5"><LevelBadge level={data.level} />{data.practices.map((p) => <span key={p} className="chip bg-white/5 text-slate-300">{p}</span>)}</div>
        {data.authored && data.segments ? <div className="mt-8"><Segments segments={data.segments} /></div> : (
          <div className="glass mt-6 p-6 text-sm text-slate-300">
            <p>This project brief is being written. It will guide you through all 14 engineering stages:</p>
            <ol className="mt-3 grid list-decimal gap-1 pl-5 sm:grid-cols-2">{STEPS.map((s) => <li key={s}>{s}</li>)}</ol>
          </div>
        )}
      </div>
    </LessonContext.Provider>
  );
}

export default function Projects() {
  const { id } = useParams();
  const { data: course } = useCourse();
  const { data: progress } = useProgress();
  if (id) return <ProjectDetail id={id} />;
  const titles = Object.fromEntries(flatLessons(course).map((l) => [l.id, l.title]));
  return (
    <div className="mx-auto max-w-6xl p-4 lg:p-8">
      <PageHeader title="Practical projects" subtitle="Progressively harder, multi-module projects modelled on real database engineering assignments: requirements → ER model → normalized schema → MySQL implementation → indexing → transactions → optimization → security → testing → production." />
      <div className="grid gap-4 md:grid-cols-2">
        {course?.projects.map((p, i) => {
          const ready = p.prerequisites.filter((x) => progress?.lessons[x]?.status === "completed").length;
          return (
            <Link key={p.id} to={`/projects/${p.id}`} className="glass block p-5 transition hover:border-white/20">
              <div className="flex items-start justify-between gap-2"><div className="flex items-center gap-2 text-xs text-slate-500"><Boxes size={14} />Project {i + 1}</div><LevelBadge level={p.level} /></div>
              <div className="mt-2 text-lg font-semibold text-white">{p.title}</div>
              <div className="mt-2 flex flex-wrap gap-1">{p.practices.map((x) => <span key={x} className="chip bg-white/5 text-slate-400">{x}</span>)}</div>
              <div className="mt-3 text-[11.5px] text-slate-500">Prerequisites ({ready}/{p.prerequisites.length} done): {p.prerequisites.map((x) => titles[x] ?? x).join(" · ")}</div>
              <div className="mt-2 text-xs">{p.authored ? <span className="inline-flex items-center gap-1 text-emerald-300"><CheckCircle2 size={12} />Brief available</span> : <span className="text-slate-500">Brief in preparation</span>}</div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
