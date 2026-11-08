import type { Segment } from "../../lib/types";
import VizBlock from "../../viz/VizBlock";
import Markdown from "../Markdown";
import ConcurrencyBlock from "./ConcurrencyBlock";
import ExerciseBlock from "./ExerciseBlock";
import QuizBlock from "./QuizBlock";
import SqlBlock from "./SqlBlock";
import WalkthroughBlock from "./WalkthroughBlock";
import { STAGE_META } from "../ui";

export default function Segments({ segments }: { segments: Segment[] }) {
  return (
    <>
      {segments.map((s, i) => {
        switch (s.type) {
          case "heading":
            return (
              <h2 key={i} id={s.anchor} data-section className="mb-4 mt-12 flex scroll-mt-24 items-center gap-3 border-t border-white/5 pt-8 text-xl font-semibold text-white first:mt-0 first:border-0 first:pt-0">
                <span className={`h-5 w-1.5 rounded-full ${STAGE_META[s.stage].color}`} />{s.title}
                <span className={`ml-auto text-[10.5px] font-medium uppercase tracking-wider ${STAGE_META[s.stage].text}`}>{STAGE_META[s.stage].label}</span>
              </h2>
            );
          case "markdown": return <Markdown key={i} content={s.content} />;
          case "sql": return <SqlBlock key={i} {...s} />;
          case "exercise": return <ExerciseBlock key={i} {...s} />;
          case "quiz": return <QuizBlock key={i} title={s.title} questions={s.questions} />;
          case "viz": return <VizBlock key={i} viz={s.viz} props={s.props} />;
          case "concurrency": return <ConcurrencyBlock key={i} {...s} />;
          case "walkthrough": return <WalkthroughBlock key={i} title={s.title} steps={s.steps} />;
          default: return null;
        }
      })}
    </>
  );
}
