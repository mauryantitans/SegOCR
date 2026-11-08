import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, HelpCircle, XCircle } from "lucide-react";
import { useState } from "react";
import { api } from "../../lib/api";
import type { Question } from "../../lib/types";
import Markdown from "../Markdown";

interface Feedback { correct: boolean; answer: number | number[]; explanation: string }

export function QuestionCard({ q, index, onAnswered, deferred, selected, onSelect, result }: {
  q: Question; index: number; onAnswered?: (ok: boolean) => void;
  deferred?: boolean; selected?: number[]; onSelect?: (s: number[]) => void; result?: Feedback | null;
}) {
  const qc = useQueryClient();
  const [local, setLocal] = useState<number[]>([]);
  const [fb, setFb] = useState<Feedback | null>(null);
  const sel = deferred ? selected ?? [] : local;
  const feedback = deferred ? result : fb;
  const setSel = (s: number[]) => (deferred ? onSelect?.(s) : setLocal(s));
  const toggle = (i: number) => {
    if (feedback) return;
    setSel(q.type === "multi" ? (sel.includes(i) ? sel.filter((x) => x !== i) : [...sel, i]) : [i]);
  };
  const check = async () => {
    const r = await api<Feedback>("/quiz/answer", { method: "POST", json: { question_id: q.id, answer: q.type === "multi" ? sel : sel[0] } });
    setFb(r); onAnswered?.(r.correct); qc.invalidateQueries({ queryKey: ["weak"] }); qc.invalidateQueries({ queryKey: ["progress"] });
  };
  const correctSet = feedback ? (Array.isArray(feedback.answer) ? feedback.answer : [feedback.answer]) : [];
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="flex gap-2 text-sm text-slate-100"><span className="text-slate-500">{index + 1}.</span><Markdown content={q.question} className="callout-body flex-1" /></div>
      {q.type === "multi" && <div className="mt-1 pl-5 text-[11px] text-slate-500">Select all that apply</div>}
      <div className="mt-3 space-y-1.5 pl-5">
        {q.options.map((o, i) => {
          const chosen = sel.includes(i);
          const right = correctSet.includes(i);
          return (
            <button key={i} onClick={() => toggle(i)} className={clsx("flex w-full items-start gap-2.5 rounded-lg border px-3 py-2 text-left text-sm transition",
              feedback ? (right ? "border-emerald-400/50 bg-emerald-500/10" : chosen ? "border-rose-400/50 bg-rose-500/10" : "border-white/5 opacity-70")
                : chosen ? "border-sky-400/60 bg-sky-500/10" : "border-white/[0.07] hover:border-white/20")}>
              <span className={clsx("mt-0.5 grid h-4 w-4 shrink-0 place-items-center border text-[10px]", q.type === "multi" ? "rounded" : "rounded-full", chosen ? "border-sky-300 bg-sky-400 text-ink-950" : "border-slate-500")}>{chosen ? "✓" : ""}</span>
              <Markdown content={o} className="callout-body flex-1" />
            </button>
          );
        })}
      </div>
      {!deferred && !feedback && <div className="mt-3 pl-5"><button className="btn-primary text-xs" disabled={!sel.length} onClick={check}>Check answer</button></div>}
      {feedback && (
        <div className={clsx("ml-5 mt-3 rounded-lg px-3 py-2 text-sm", feedback.correct ? "bg-emerald-500/10 text-emerald-100" : "bg-rose-500/10 text-rose-100")}>
          <div className="mb-1 flex items-center gap-1.5 font-medium">{feedback.correct ? <><CheckCircle2 size={15} />Correct</> : <><XCircle size={15} />Not quite</>}</div>
          <Markdown content={feedback.explanation} className="callout-body" />
        </div>
      )}
    </div>
  );
}

export default function QuizBlock({ title, questions }: { title?: string; questions: Question[] }) {
  const [score, setScore] = useState<Record<string, boolean>>({});
  const answered = Object.keys(score).length;
  return (
    <div className="not-prose my-6 rounded-xl border border-rose-300/15 bg-ink-900/40">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-2.5">
        <div className="flex items-center gap-2"><HelpCircle size={15} className="text-rose-300" /><span className="text-sm font-semibold text-slate-100">{title ?? "Knowledge check"}</span></div>
        <span className="text-xs text-slate-500">{answered ? `${Object.values(score).filter(Boolean).length}/${answered} correct` : `${questions.length} questions`}</span>
      </div>
      <div className="space-y-3 p-4">
        {questions.map((q, i) => <QuestionCard key={q.id} q={q} index={i} onAnswered={(ok) => setScore((s) => ({ ...s, [q.id]: ok }))} />)}
      </div>
    </div>
  );
}
