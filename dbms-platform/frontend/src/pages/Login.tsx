import { Database } from "lucide-react";
import { useState, type FormEvent } from "react";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setErr(null); setBusy(true);
    try { mode === "login" ? await login(username, password) : await register(username, password, displayName || undefined); }
    catch (e2) { setErr((e2 as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className="grid min-h-full place-items-center p-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-sky-400 to-indigo-500 text-ink-950"><Database size={28} /></div>
          <h1 className="text-2xl font-semibold text-white">DB Academy</h1>
          <p className="mt-2 text-sm text-slate-400">Database systems and MySQL, from relational theory to production engineering — with a real MySQL sandbox.</p>
        </div>
        <form onSubmit={submit} className="glass space-y-4 p-6">
          <div className="flex rounded-lg bg-white/5 p-1 text-sm">
            {(["login", "register"] as const).map((m) => (
              <button type="button" key={m} onClick={() => setMode(m)} className={`flex-1 rounded-md py-1.5 ${mode === m ? "bg-sky-500/20 text-sky-200" : "text-slate-400"}`}>
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>
          <label className="block text-sm"><span className="text-slate-400">Username</span><input className="input mt-1" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required /></label>
          {mode === "register" && <label className="block text-sm"><span className="text-slate-400">Display name (optional)</span><input className="input mt-1" value={displayName} onChange={(e) => setDisplayName(e.target.value)} /></label>}
          <label className="block text-sm"><span className="text-slate-400">Password {mode === "register" && "(min 8 characters)"}</span><input className="input mt-1" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} required /></label>
          {err && <div className="rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{err}</div>}
          <button className="btn-primary w-full justify-center py-2" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
        </form>
      </div>
    </div>
  );
}
