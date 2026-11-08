import { createContext, useContext } from "react";

export interface LessonCtx {
  docId: string;
  dataset: string;
  /** SQL of every runnable block in document order, used for "replay from start". */
  runnable: { id: string; sql: string }[];
  captureVersion?: string;
}
export const LessonContext = createContext<LessonCtx>({ docId: "", dataset: "empty", runnable: [] });
export const useLessonCtx = () => useContext(LessonContext);
