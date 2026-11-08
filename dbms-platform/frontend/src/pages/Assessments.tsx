import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Award, ClipboardCheck, MessageSquareQuote } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Loading } from "../App";
import ExerciseBlock from "../components/lesson/ExerciseBlock";
import { QuestionCard } from "../components/lesson/QuizBlock";
import Markdown from "../components/Markdown";
import { PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { useCourse, useProgress } from "../lib/queries";
import type { ExerciseDef, Question } from "../lib/types";

interface Paper { id: string; title: string; description?: string; passMark: number; questions: Question[]; challenges?: ExerciseDef[]; interview?: { id: string; question: string }[]; available?: number }
interface Result { score: number; passed: boolean; correct: number; total: number; results: { questionId: string; correct: boolean; answer: number | number[]; explanation: string }[] }

function Interview({ q }: { q: { id: string; question: string } }) {
  const [ans, setAns] = useState<string | null>(null);
  const [mine, setMine] = useState("");
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="flex gap-2 text-sm text-slate-100"><MessageSquareQuote size={16} className="mt-0.5 shrink-0 text-violet-300" /><Markdown content={q.question} className="callout-body" /></div>
      <textarea className="input mt-3 min-h-24" placeholder="Answer as you would in an interview, then compare." value={mine} onChange={(e) => setMine(e.target.value)} />
      {ans === null ? <button className="btn-outline mt-2 text-xs" disabled={mine.trim().length < 30} onClick={async () => setAns((await api<{ answer: string }>(`/interview/${q.id}/answer`)).answer)}>Compare with model answer</button>
        : <div className="mt-3 rounded-lg bg-violet-500/[0.07] p-3"><div className="mb-1 text-xs font-semibold text-violet-300">Model answer</div><Markdown content={ans} className="callout-body" /></div>}
    </div>
  );
}

function PaperView({ kind, id }: { kind: string; id: string }) {
  const qc = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["paper", kind, id], queryFn: () => api<Paper>(kind === "exam" ? `/exams/${id}` : `/assessments/${id}`) });
  const [answers, setAnswers] = useState<Record<string, number[]>>({});
  const [result, setResult] = useState<Result | null>(null);
  if (isLoading) return <Loading />;
  if (error || !data) return <div className="p-10 text-slate-400">This assessment is not available yet. <Link to="/assessments" className="text-sky-300">Back</Link></div>;
  const submit = async () => {
    const payload = Object.fromEntries(data.questions.map((q) => [q.id, q.type === "multi" ? answers[q.id] ?? [] : (answers[q.id] ?? [])[0]]).filter(([, v]) => v !== undefined));
    const r = await api<Result>(kind === "exam" ? `/exams/${id}/submit` : `/assessments/${id}/submit`, { method: "POST", json: { answers: payload } });
    setResult(r); qc.invalidateQueries({ queryKey: ["progress"] }); qc.invalidateQueries({ queryKey: ["weak"] });
    document.querySelector("main")?.scrollTo({ top: 0, behavior: "smooth" });
  };
  const byId = Object.fromEntries((result?.results ?? []).map((r) => [r.questionId, r]));
  return (
    <div className="mx-auto max-w-4xl p-4 lg:p-8">
      <Link to="/assessments" className="text-xs text-slate-500">← Assessments</Link>
      <h1 className="mt-2 text-2xl font-semibold text-white">{data.title}</h1>
      {data.description && <p className="mt-1 text-sm text-slate-400">{data.description}</p>}
      <p className="mt-1 text-xs text-slate-500">{data.questions.length} questions · pass mark {data.passMark}%{kind === "exam" ? " · questions are drawn from the module question banks (the selection changes daily)" : ""}</p>
      {result && (
        <div className={clsx("mt-4 rounded-xl border p-4", result.passed ? "border-emerald-400/30 bg-emerald-500/10" : "border-amber-400/30 bg-amber-500/10")}>
          <div className="flex items-center gap-2 text-lg font-semibold text-white"><Award size={18} />{result.score}% — {result.passed ? "Passed" : "Not yet passed"}</div>
          <div className="text-sm text-slate-300">{result.correct} of {result.total} correct. Explanations are shown under each question; missed concepts now feed your revision list on the dashboard.</div>
        </div>
      )}
      <div className="mt-6 space-y-3">
        {data.questions.map((q, i) => (
          <QuestionCard key={q.id} q={q} index={i} deferred selected={answers[q.id]} onSelect={(s) => setAnswers((a) => ({ ...a, [q.id]: s }))}
            result={byId[q.id] ? { correct: byId[q.id].correct, answer: byId[q.id].answer, explanation: byId[q.id].explanation } : null} />
        ))}
      </div>
      {!result && data.questions.length > 0 && <button className="btn-primary mt-4" onClick={submit}>Submit answers ({Object.keys(answers).length}/{data.questions.length} answered)</button>}
      {!!data.challenges?.length && <><h2 className="mt-10 text-lg font-semibold text-white">SQL challenges</h2>{data.challenges.map((c) => <ExerciseBlock key={c.id} {...c} />)}</>}
      {!!data.interview?.length && <><h2 className="mt-10 mb-3 text-lg font-semibold text-white">Interview questions</h2><div className="space-y-3">{data.interview.map((q) => <Interview key={q.id} q={q} />)}</div></>}
    </div>
  );
}

export default function Assessments() {
  const { kind, id } = useParams();
  const { data: course } = useCourse();
  const { data: progress } = useProgress();
  if (kind && id) return <PaperView kind={kind} id={id} />;
  return (
    <div className="mx-auto max-w-5xl p-4 lg:p-8">
      <PageHeader title="Assessments" subtitle="Module assessments (question bank, graded SQL challenges and interview questions) and cumulative exams across modules." />
      <h2 className="mb-3 text-sm font-semibold text-slate-300">Cumulative exams</h2>
      <div className="grid gap-3 md:grid-cols-3">
        {course?.exams.map((e) => {
          const best = progress?.assessments[e.id];
          return <Link key={e.id} to={`/assessments/exam/${e.id}`} className="glass block p-4 hover:border-white/20"><ClipboardCheck size={16} className="text-sky-300" /><div className="mt-2 font-medium text-white">{e.title}</div><div className="mt-1 text-xs text-slate-500">{e.questions} questions{best ? ` · best ${best.score}%` : ""}</div></Link>;
        })}
      </div>
      <h2 className="mb-3 mt-8 text-sm font-semibold text-slate-300">Module assessments</h2>
      <div className="divide-y divide-white/5 rounded-xl border border-white/5">
        {course?.modules.map((m) => {
          const best = progress?.assessments[`${m.id}-assessment`];
          return (
            <div key={m.id} className="flex items-center gap-3 px-4 py-3">
              <span className="w-8 text-sm text-slate-500">{m.number}</span><span className="flex-1 text-sm text-slate-200">{m.title}</span>
              {best && <span className={clsx("chip", best.passed ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300")}>best {best.score}%</span>}
              {m.hasAssessment ? <Link className="btn-outline text-xs" to={`/assessments/module/${m.id}-assessment`}>Open</Link> : <span className="text-xs text-slate-600">in preparation</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
