import { Plus, Terminal, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CodeBlock, Mermaid } from "../components/Markdown";

interface Attr { name: string; type: string; pk: boolean; nullable: boolean }
interface Entity { name: string; attrs: Attr[] }
interface Rel { from: string; to: string; card: "1:1" | "1:N" | "M:N"; label: string; total: boolean }

const START: { entities: Entity[]; rels: Rel[] } = {
  entities: [
    { name: "student", attrs: [{ name: "student_id", type: "INT UNSIGNED", pk: true, nullable: false }, { name: "name", type: "VARCHAR(80)", pk: false, nullable: false }, { name: "email", type: "VARCHAR(255)", pk: false, nullable: false }] },
    { name: "course", attrs: [{ name: "course_id", type: "VARCHAR(10)", pk: true, nullable: false }, { name: "title", type: "VARCHAR(100)", pk: false, nullable: false }] },
    { name: "department", attrs: [{ name: "dept_id", type: "SMALLINT UNSIGNED", pk: true, nullable: false }, { name: "name", type: "VARCHAR(60)", pk: false, nullable: false }] },
  ],
  rels: [
    { from: "student", to: "course", card: "M:N", label: "enrolls in", total: false },
    { from: "department", to: "course", card: "1:N", label: "offers", total: true },
  ],
};

const pkOf = (e: Entity) => e.attrs.filter((a) => a.pk);
const ident = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9_]+/g, "_");

export function generate(entities: Entity[], rels: Rel[]) {
  const extraCols: Record<string, string[]> = {};
  const fks: Record<string, string[]> = {};
  const junctions: string[] = [];
  const byName = Object.fromEntries(entities.map((e) => [e.name, e]));
  for (const r of rels) {
    const a = byName[r.from], b = byName[r.to];
    if (!a || !b) continue;
    if (r.card === "M:N") {
      const t = `${a.name}_${b.name}`;
      const cols = [...pkOf(a).map((k) => ({ ...k, ref: a })), ...pkOf(b).map((k) => ({ ...k, ref: b }))];
      const lines = cols.map((c) => `  ${c.ref.name}_${c.name} ${c.type} NOT NULL`);
      lines.push(`  PRIMARY KEY (${cols.map((c) => `${c.ref.name}_${c.name}`).join(", ")})`);
      lines.push(`  CONSTRAINT fk_${t}_${a.name} FOREIGN KEY (${pkOf(a).map((k) => `${a.name}_${k.name}`).join(", ")}) REFERENCES ${a.name} (${pkOf(a).map((k) => k.name).join(", ")})`);
      lines.push(`  CONSTRAINT fk_${t}_${b.name} FOREIGN KEY (${pkOf(b).map((k) => `${b.name}_${k.name}`).join(", ")}) REFERENCES ${b.name} (${pkOf(b).map((k) => k.name).join(", ")})`);
      junctions.push(`-- M:N "${r.label}" becomes a junction table\nCREATE TABLE ${t} (\n${lines.join(",\n")}\n) ENGINE=InnoDB;`);
    } else {
      // FK goes on the "many" side (b) for 1:N; for 1:1 on b with a UNIQUE constraint.
      const cols = pkOf(a).map((k) => `${a.name}_${k.name}`);
      extraCols[b.name] = [...(extraCols[b.name] ?? []), ...pkOf(a).map((k) => `  ${a.name}_${k.name} ${k.type} ${r.total ? "NOT NULL" : "NULL"}`)];
      fks[b.name] = [...(fks[b.name] ?? []),
        ...(r.card === "1:1" ? [`  UNIQUE KEY uq_${b.name}_${a.name} (${cols.join(", ")})`] : [`  KEY idx_${b.name}_${a.name} (${cols.join(", ")})`]),
        `  CONSTRAINT fk_${b.name}_${a.name} FOREIGN KEY (${cols.join(", ")}) REFERENCES ${a.name} (${pkOf(a).map((k) => k.name).join(", ")})`];
    }
  }
  const order = [...entities].sort((x, y) => (fks[x.name] ? 1 : 0) - (fks[y.name] ? 1 : 0));
  const tables = order.map((e) => {
    const lines = e.attrs.map((a) => `  ${a.name} ${a.type} ${a.nullable && !a.pk ? "NULL" : "NOT NULL"}`);
    lines.push(...(extraCols[e.name] ?? []));
    if (pkOf(e).length) lines.push(`  PRIMARY KEY (${pkOf(e).map((k) => k.name).join(", ")})`);
    lines.push(...(fks[e.name] ?? []));
    return `CREATE TABLE ${e.name} (\n${lines.join(",\n")}\n) ENGINE=InnoDB;`;
  });
  const card = (r: Rel) => (r.card === "1:1" ? "||--o|" : r.card === "1:N" ? (r.total ? "||--|{" : "||--o{") : "}o--o{");
  const mermaid = ["erDiagram",
    ...entities.map((e) => `  ${e.name} {\n${e.attrs.map((a) => `    ${a.type.split("(")[0].replace(/\s+/g, "_")} ${a.name}${a.pk ? " PK" : ""}`).join("\n")}\n  }`),
    ...rels.filter((r) => byName[r.from] && byName[r.to]).map((r) => `  ${r.from} ${card(r)} ${r.to} : "${r.label.replace(/"/g, "")}"`)].join("\n");
  return { ddl: [...tables, ...junctions].join("\n\n"), mermaid };
}

export default function ErBuilder() {
  const [entities, setEntities] = useState<Entity[]>(START.entities);
  const [rels, setRels] = useState<Rel[]>(START.rels);
  const nav = useNavigate();
  const out = useMemo(() => generate(entities, rels), [entities, rels]);
  const upd = (i: number, f: (e: Entity) => Entity) => setEntities((es) => es.map((e, k) => (k === i ? f(e) : e)));

  return (
    <div className="not-prose my-5 rounded-xl border border-violet-400/20 bg-ink-900/60 p-4 text-sm">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-violet-300">ER diagram builder → MySQL schema</div>
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-3">
          {entities.map((e, i) => (
            <div key={i} className="rounded-lg border border-white/10 p-3">
              <div className="flex items-center gap-2">
                <input className="input py-1 font-mono font-semibold" value={e.name} onChange={(ev) => upd(i, (x) => ({ ...x, name: ident(ev.target.value) }))} />
                <button className="btn-ghost px-1.5" onClick={() => setEntities((es) => es.filter((_, k) => k !== i))} title="Remove entity"><Trash2 size={14} /></button>
              </div>
              <div className="mt-2 space-y-1">
                {e.attrs.map((a, j) => (
                  <div key={j} className="flex items-center gap-1.5">
                    <input className="input py-0.5 font-mono text-xs" value={a.name} onChange={(ev) => upd(i, (x) => ({ ...x, attrs: x.attrs.map((y, k) => (k === j ? { ...y, name: ident(ev.target.value) } : y)) }))} />
                    <input className="input w-40 py-0.5 font-mono text-xs" value={a.type} onChange={(ev) => upd(i, (x) => ({ ...x, attrs: x.attrs.map((y, k) => (k === j ? { ...y, type: ev.target.value.toUpperCase() } : y)) }))} />
                    <label className="flex items-center gap-1 text-[11px] text-slate-400"><input type="checkbox" checked={a.pk} onChange={(ev) => upd(i, (x) => ({ ...x, attrs: x.attrs.map((y, k) => (k === j ? { ...y, pk: ev.target.checked } : y)) }))} />PK</label>
                    <label className="flex items-center gap-1 text-[11px] text-slate-400"><input type="checkbox" checked={a.nullable} onChange={(ev) => upd(i, (x) => ({ ...x, attrs: x.attrs.map((y, k) => (k === j ? { ...y, nullable: ev.target.checked } : y)) }))} />NULL</label>
                    <button className="btn-ghost px-1" onClick={() => upd(i, (x) => ({ ...x, attrs: x.attrs.filter((_, k) => k !== j) }))}><Trash2 size={12} /></button>
                  </div>
                ))}
                <button className="btn-ghost px-1.5 py-0.5 text-xs" onClick={() => upd(i, (x) => ({ ...x, attrs: [...x.attrs, { name: "new_column", type: "VARCHAR(100)", pk: false, nullable: true }] }))}><Plus size={12} />attribute</button>
              </div>
            </div>
          ))}
          <button className="btn-outline text-xs" onClick={() => setEntities((es) => [...es, { name: `entity_${es.length + 1}`, attrs: [{ name: "id", type: "INT UNSIGNED", pk: true, nullable: false }] }])}><Plus size={12} />Add entity</button>
          <div className="rounded-lg border border-white/10 p-3">
            <div className="mb-2 text-xs text-slate-400">Relationships (for 1:N the first entity is the "one" side)</div>
            {rels.map((r, i) => (
              <div key={i} className="mb-1 flex flex-wrap items-center gap-1.5">
                {(["from", "to"] as const).map((side) => (
                  <select key={side} className="rounded bg-ink-800 px-1.5 py-1 font-mono text-xs" value={r[side]} onChange={(ev) => setRels((rs) => rs.map((x, k) => (k === i ? { ...x, [side]: ev.target.value } : x)))}>
                    {entities.map((e) => <option key={e.name}>{e.name}</option>)}</select>))}
                <select className="rounded bg-ink-800 px-1.5 py-1 text-xs" value={r.card} onChange={(ev) => setRels((rs) => rs.map((x, k) => (k === i ? { ...x, card: ev.target.value as Rel["card"] } : x)))}>{["1:1", "1:N", "M:N"].map((c) => <option key={c}>{c}</option>)}</select>
                <input className="input w-32 py-0.5 text-xs" value={r.label} onChange={(ev) => setRels((rs) => rs.map((x, k) => (k === i ? { ...x, label: ev.target.value } : x)))} />
                <label className="flex items-center gap-1 text-[11px] text-slate-400" title="Total participation of the many side → NOT NULL foreign key"><input type="checkbox" checked={r.total} onChange={(ev) => setRels((rs) => rs.map((x, k) => (k === i ? { ...x, total: ev.target.checked } : x)))} />mandatory</label>
                <button className="btn-ghost px-1" onClick={() => setRels((rs) => rs.filter((_, k) => k !== i))}><Trash2 size={12} /></button>
              </div>
            ))}
            {entities.length >= 2 && <button className="btn-ghost px-1.5 py-0.5 text-xs" onClick={() => setRels((rs) => [...rs, { from: entities[0].name, to: entities[1].name, card: "1:N", label: "relates to", total: false }])}><Plus size={12} />relationship</button>}
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <Mermaid code={out.mermaid} />
          <div className="flex items-center justify-between"><span className="text-xs text-slate-400">Generated MySQL DDL</span>
            <button className="btn-primary text-xs" onClick={() => nav("/playground", { state: { sql: out.ddl + "\n\nSHOW TABLES;", dataset: "empty" } })}><Terminal size={12} />Run in playground</button></div>
          <CodeBlock code={out.ddl} lang="sql" />
          <p className="text-[11px] text-slate-500">Mapping rules: 1:N → foreign key on the many side (NOT NULL if participation is mandatory) plus an index; 1:1 → foreign key with a UNIQUE constraint; M:N → junction table whose primary key combines both foreign keys.</p>
        </div>
      </div>
    </div>
  );
}
