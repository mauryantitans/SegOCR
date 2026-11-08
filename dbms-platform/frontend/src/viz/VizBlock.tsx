import BPlusTreeViz from "./BPlusTreeViz";
import BufferPoolViz from "./BufferPoolViz";
import ErBuilder from "./ErBuilder";
import FdAnalyzer from "./FdAnalyzer";
import IsolationMatrix from "./IsolationMatrix";
import JoinViz from "./JoinViz";
import QueryOrderViz from "./QueryOrderViz";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const VIZ: Record<string, React.ComponentType<any>> = {
  bptree: BPlusTreeViz, join: JoinViz, fd: FdAnalyzer, bufferpool: BufferPoolViz, "query-order": QueryOrderViz,
  isolation: IsolationMatrix, "er-builder": ErBuilder,
};

export default function VizBlock({ viz, props }: { viz: string; props: Record<string, unknown> }) {
  const C = VIZ[viz];
  if (!C) return <div className="my-4 rounded-lg bg-amber-500/10 p-3 text-sm text-amber-200">Unknown visualization “{viz}”.</div>;
  return <C {...props} />;
}
