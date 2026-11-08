import clsx from "clsx";
import { Link, useParams } from "react-router-dom";
import { PageHeader } from "../components/ui";
import { VIZ } from "../viz/VizBlock";

const ITEMS = [
  { id: "bptree", title: "B+ tree index", desc: "Insert keys, watch leaf and internal splits, trace searches and range scans across linked leaves.", lesson: "m07-btree-bplustree" },
  { id: "join", title: "Join operations", desc: "Step through INNER, LEFT, RIGHT, FULL (emulated) and CROSS joins row by row, including NULL-extended rows.", lesson: "m04-outer-joins" },
  { id: "er-builder", title: "ER diagram builder", desc: "Model entities and relationships, see the ER diagram and the generated MySQL DDL, run it in the playground.", lesson: "m05-er-to-relational" },
  { id: "fd", title: "Normalization analyzer", desc: "Attribute closures, candidate keys, normal form violations, minimal cover, 3NF synthesis and BCNF decomposition.", lesson: "m06-closure-keys" },
  { id: "query-order", title: "Logical query order", desc: "How SQL clauses are written vs logically evaluated.", lesson: "m04-logical-query-order" },
  { id: "isolation", title: "Isolation levels & anomalies", desc: "Which anomalies each isolation level allows — SQL standard vs InnoDB behaviour.", lesson: "m08-isolation-levels" },
  { id: "bufferpool", title: "Buffer pool LRU", desc: "InnoDB's midpoint-insertion LRU with young and old sublists.", lesson: "m10-buffer-pool" },
];

export default function Visualizations() {
  const { viz = "bptree" } = useParams();
  const item = ITEMS.find((i) => i.id === viz) ?? ITEMS[0];
  const C = VIZ[item.id];
  return (
    <div className="mx-auto max-w-7xl p-4 lg:p-8">
      <PageHeader title="Interactive visualizations" subtitle="Algorithmic visualizations computed in your browser. Execution plans, locks and concurrency are shown live from MySQL in the Playground and Concurrency Lab." />
      <div className="mb-5 flex flex-wrap gap-2">
        {ITEMS.map((i) => <Link key={i.id} to={`/visualizations/${i.id}`} className={clsx("rounded-lg border px-3 py-1.5 text-sm", i.id === item.id ? "border-sky-400/50 bg-sky-500/10 text-sky-200" : "border-white/10 text-slate-400 hover:text-slate-200")}>{i.title}</Link>)}
      </div>
      <p className="text-sm text-slate-400">{item.desc} <Link to={`/lessons/${item.lesson}`} className="text-sky-300">Related lesson →</Link></p>
      <C />
    </div>
  );
}
