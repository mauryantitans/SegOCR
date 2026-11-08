import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { useAuth } from "./lib/auth";
import Curriculum from "./pages/Curriculum";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";

const Lesson = lazy(() => import("./pages/Lesson"));
const Playground = lazy(() => import("./pages/Playground"));
const ConcurrencyLab = lazy(() => import("./pages/ConcurrencyLab"));
const Visualizations = lazy(() => import("./pages/Visualizations"));
const Projects = lazy(() => import("./pages/Projects"));
const Assessments = lazy(() => import("./pages/Assessments"));
const Review = lazy(() => import("./pages/Review"));

export function Loading() {
  return <div className="p-10 text-sm text-slate-500">Loading…</div>;
}

export default function App() {
  const { user, ready } = useAuth();
  if (!ready) return <Loading />;
  if (!user) return <Login />;
  return (
    <Layout>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/curriculum" element={<Curriculum />} />
          <Route path="/lessons/:id" element={<Lesson />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/concurrency" element={<ConcurrencyLab />} />
          <Route path="/visualizations" element={<Visualizations />} />
          <Route path="/visualizations/:viz" element={<Visualizations />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<Projects />} />
          <Route path="/assessments" element={<Assessments />} />
          <Route path="/assessments/:kind/:id" element={<Assessments />} />
          <Route path="/review" element={<Review />} />
          <Route path="*" element={<div className="p-10 text-slate-400">Page not found.</div>} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
