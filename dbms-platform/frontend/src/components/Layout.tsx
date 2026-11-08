import clsx from "clsx";
import { BookOpen, Boxes, ClipboardCheck, Database, FlaskConical, GitBranch, LayoutDashboard, LogOut, Menu, NotebookPen, Search, Terminal, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useProgress } from "../lib/queries";
import type { LessonSummary } from "../lib/types";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/curriculum", label: "Curriculum", icon: BookOpen },
  { to: "/playground", label: "SQL Playground", icon: Terminal },
  { to: "/concurrency", label: "Concurrency Lab", icon: GitBranch },
  { to: "/visualizations", label: "Visualizations", icon: FlaskConical },
  { to: "/projects", label: "Projects", icon: Boxes },
  { to: "/assessments", label: "Assessments", icon: ClipboardCheck },
  { to: "/review", label: "Review & Notes", icon: NotebookPen },
];

function SearchBox() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<LessonSummary[]>([]);
  const nav = useNavigate();
  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const t = setTimeout(() => api<LessonSummary[]>(`/search?q=${encodeURIComponent(q)}`).then(setHits).catch(() => setHits([])), 200);
    return () => clearTimeout(t);
  }, [q]);
  return (
    <div className="relative w-full max-w-md">
      <Search size={15} className="absolute left-3 top-2.5 text-slate-500" />
      <input className="input pl-9" placeholder="Search lessons, topics, concepts…" value={q} onChange={(e) => setQ(e.target.value)} />
      {hits.length > 0 && (
        <div className="glass-strong absolute z-50 mt-1 max-h-96 w-full overflow-auto p-1 scrollbar-thin">
          {hits.map((h) => (
            <button key={h.id} className="block w-full rounded-lg px-3 py-2 text-left hover:bg-white/5"
              onClick={() => { setQ(""); nav(`/lessons/${h.id}`); }}>
              <div className="text-sm text-slate-100">{h.title}</div>
              <div className="text-[11px] text-slate-500">Module {h.moduleId.slice(1)} · {h.level}{h.status === "planned" ? " · planned" : ""}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const { data: progress } = useProgress();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);
  const pct = progress && progress.stats.authoredLessons ? Math.round((100 * progress.stats.completedLessons) / progress.stats.authoredLessons) : 0;

  return (
    <div className="flex h-full">
      <aside className={clsx("fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-white/5 bg-ink-900/95 backdrop-blur-xl transition-transform lg:static lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full")}>
        <Link to="/" className="flex items-center gap-2.5 px-5 py-5">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-sky-400 to-indigo-500 text-ink-950"><Database size={19} /></div>
          <div><div className="text-[15px] font-semibold text-white">DB Academy</div><div className="text-[11px] text-slate-500">DBMS · MySQL 8.4 LTS</div></div>
        </Link>
        <nav className="flex-1 space-y-0.5 px-3">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"}
              className={({ isActive }) => clsx("flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition",
                isActive ? "bg-sky-500/10 text-sky-300" : "text-slate-400 hover:bg-white/5 hover:text-slate-200")}>
              <Icon size={17} />{label}
            </NavLink>
          ))}
        </nav>
        <div className="m-3 rounded-xl border border-white/5 bg-white/[0.03] p-3">
          <div className="flex justify-between text-xs text-slate-400"><span>Course progress</span><span>{pct}%</span></div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5"><div className="h-full rounded-full bg-gradient-to-r from-sky-400 to-indigo-400" style={{ width: `${pct}%` }} /></div>
          <div className="mt-3 flex items-center justify-between">
            <div className="truncate text-xs text-slate-300">{user?.displayName}</div>
            <button className="btn-ghost px-1.5 py-1" title="Sign out" onClick={logout}><LogOut size={14} /></button>
          </div>
        </div>
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/5 bg-ink-950/70 px-4 py-3 backdrop-blur-xl lg:px-8">
          <button className="btn-ghost px-2 lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu">{open ? <X size={18} /> : <Menu size={18} />}</button>
          <SearchBox />
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
