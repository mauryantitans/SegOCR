import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import ini from "highlight.js/lib/languages/ini";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import yaml from "highlight.js/lib/languages/yaml";
import javascript from "highlight.js/lib/languages/javascript";
import type { Root } from "mdast";
import { memo, useEffect, useId, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";
import remarkDirective from "remark-directive";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";

hljs.registerLanguage("sql", sql); hljs.registerLanguage("bash", bash); hljs.registerLanguage("json", json);
hljs.registerLanguage("python", python); hljs.registerLanguage("yaml", yaml); hljs.registerLanguage("ini", ini);
hljs.registerLanguage("javascript", javascript);

const TITLES: Record<string, string> = {
  theory: "DBMS Theory", mysql: "MySQL Implementation", internals: "Internal Working", example: "Example",
  "best-practice": "Best Practice", warning: "Warning", performance: "Performance", industry: "Industry Practice",
  note: "Note", standard: "Standard SQL vs MySQL", version: "Version Note",
};

function textOf(node: any): string {
  if (node.value) return node.value;
  return (node.children ?? []).map(textOf).join("");
}

/** Turns :::name[title] container directives into styled callouts; restores accidental text directives. */
function remarkCallouts() {
  return (tree: Root) => {
    visit(tree, (node: any, index, parent: any) => {
      if (node.type === "containerDirective") {
        let title = TITLES[node.name] ?? node.name;
        const first = node.children[0];
        if (first?.data?.directiveLabel) { title = textOf(first); node.children.shift(); }
        node.data = { hName: "div", hProperties: { className: `callout callout-${node.name} not-prose` } };
        node.children = [
          { type: "calloutTitle", data: { hName: "div", hProperties: { className: "callout-title" } }, children: [{ type: "text", value: title }] },
          { type: "calloutBody", data: { hName: "div", hProperties: { className: PROSE_SM } }, children: node.children },
        ];
      } else if ((node.type === "textDirective" || node.type === "leafDirective") && parent && index != null) {
        const text = `:${node.name}${node.children?.length ? `[${textOf(node)}]` : ""}`;
        parent.children.splice(index, 1, node.type === "leafDirective" ? { type: "paragraph", children: [{ type: "text", value: ":" + text }] } : { type: "text", value: text });
      }
    });
  };
}

let mermaidPromise: Promise<typeof import("mermaid")["default"]> | null = null;
function loadMermaid() {
  mermaidPromise ??= import("mermaid").then((m) => {
    m.default.initialize({
      startOnLoad: false, theme: "dark", securityLevel: "strict", fontFamily: "Inter, sans-serif",
      themeVariables: { background: "#0b1120", primaryColor: "#1e293b", primaryBorderColor: "#38bdf8", lineColor: "#64748b", primaryTextColor: "#e2e8f0" },
    });
    return m.default;
  });
  return mermaidPromise;
}

export function Mermaid({ code }: { code: string }) {
  const id = "m" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const [svg, setSvg] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    loadMermaid().then((m) => m.render(id, code)).then((r) => alive && setSvg(r.svg)).catch((e) => alive && setErr(String(e?.message ?? e)));
    return () => { alive = false; };
  }, [code, id]);
  if (err) return <pre className="text-xs text-rose-300">Diagram error: {err}{"\n"}{code}</pre>;
  return <div className="not-prose my-5 flex justify-center overflow-x-auto rounded-xl border border-white/5 bg-ink-900/60 p-4" dangerouslySetInnerHTML={{ __html: svg }} />;
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const html = lang && hljs.getLanguage(lang) ? hljs.highlight(code, { language: lang }).value : hljs.highlightAuto(code).value;
  return <pre className="not-prose my-4 overflow-x-auto rounded-xl border border-white/5 bg-ink-900 p-4 font-mono text-[13px] leading-relaxed scrollbar-thin"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>;
}

export const PROSE = "md prose prose-invert max-w-none prose-headings:scroll-mt-24 prose-a:text-sky-300 prose-pre:bg-ink-900 prose-th:text-slate-200 prose-table:text-sm";
export const PROSE_SM = "md prose prose-invert prose-sm max-w-none prose-p:my-1.5 prose-li:my-0.5 prose-a:text-sky-300";

function Markdown({ content, className = "lesson-prose" }: { content: string; className?: string }) {
  const cls = className.replace("lesson-prose", PROSE).replace("callout-body", PROSE_SM);
  return (
    <div className={cls}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkDirective, remarkCallouts]}
        components={{
          code({ className: cls, children }) {
            const lang = /language-([\w-]+)/.exec(cls ?? "")?.[1];
            const text = String(children).replace(/\n$/, "");
            if (lang === "mermaid") return <Mermaid code={text} />;
            if (lang || text.includes("\n")) return <CodeBlock code={text} lang={lang} />;
            return <code>{children}</code>;
          },
          pre: ({ children }) => <>{children}</>,
          a({ href, children }) {
            if (href?.startsWith("/")) return <Link to={href}>{children}</Link>;
            return <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>;
          },
        }}
      >{content}</ReactMarkdown>
    </div>
  );
}
export default memo(Markdown);
