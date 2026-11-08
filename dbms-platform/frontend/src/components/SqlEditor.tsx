import Editor, { loader, type OnMount } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import { useRef } from "react";

// Bundle Monaco locally (no CDN) and register a theme matching the app.
(self as unknown as { MonacoEnvironment: unknown }).MonacoEnvironment = { getWorker: () => new editorWorker() };
loader.config({ monaco });
monaco.editor.defineTheme("dbacademy", {
  base: "vs-dark", inherit: true,
  rules: [
    { token: "keyword", foreground: "7dd3fc", fontStyle: "bold" },
    { token: "string", foreground: "86efac" },
    { token: "number", foreground: "fca5a5" },
    { token: "comment", foreground: "64748b", fontStyle: "italic" },
    { token: "operator", foreground: "f9a8d4" },
    { token: "predefined", foreground: "c4b5fd" },
  ],
  colors: { "editor.background": "#0b1120", "editor.lineHighlightBackground": "#131c31", "editorGutter.background": "#0b1120" },
});

const KEYWORDS = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET", "JOIN", "LEFT JOIN", "RIGHT JOIN",
  "INNER JOIN", "CROSS JOIN", "ON", "USING", "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE FROM", "CREATE TABLE", "ALTER TABLE",
  "DROP TABLE", "CREATE INDEX", "PRIMARY KEY", "FOREIGN KEY", "REFERENCES", "WITH", "RECURSIVE", "UNION ALL", "EXISTS", "NOT EXISTS",
  "CASE WHEN", "THEN", "ELSE", "END", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "OVER", "PARTITION BY",
  "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "EXPLAIN", "EXPLAIN ANALYZE", "START TRANSACTION", "COMMIT", "ROLLBACK",
  "SAVEPOINT", "FOR UPDATE", "FOR SHARE", "NOWAIT", "SKIP LOCKED"];
let completionRegistered = false;
let schemaWords: string[] = [];
export function setEditorSchemaWords(words: string[]) { schemaWords = words; }

function registerCompletion() {
  if (completionRegistered) return;
  completionRegistered = true;
  monaco.languages.registerCompletionItemProvider("mysql", {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = { startLineNumber: position.lineNumber, endLineNumber: position.lineNumber, startColumn: word.startColumn, endColumn: word.endColumn };
      return {
        suggestions: [
          ...KEYWORDS.map((k) => ({ label: k, kind: monaco.languages.CompletionItemKind.Keyword, insertText: k, range })),
          ...schemaWords.map((w) => ({ label: w, kind: monaco.languages.CompletionItemKind.Field, insertText: w, range })),
        ],
      };
    },
  });
}

interface Props {
  value: string; onChange: (v: string) => void; onRun?: (sqlOrSelection: string) => void;
  height?: number | string; readOnly?: boolean;
}

export default function SqlEditor({ value, onChange, onRun, height = 220, readOnly }: Props) {
  const runRef = useRef(onRun);
  runRef.current = onRun;
  const mount: OnMount = (editor) => {
    registerCompletion();
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      const sel = editor.getSelection();
      const text = sel && !sel.isEmpty() ? editor.getModel()!.getValueInRange(sel) : editor.getValue();
      runRef.current?.(text);
    });
  };
  return (
    <div className="overflow-hidden rounded-xl border border-white/10">
      <Editor
        height={height} language="mysql" theme="dbacademy" value={value} onChange={(v) => onChange(v ?? "")} onMount={mount}
        options={{
          minimap: { enabled: false }, fontSize: 13.5, fontFamily: "JetBrains Mono, ui-monospace, monospace", scrollBeyondLastLine: false,
          wordWrap: "on", tabSize: 2, readOnly, automaticLayout: true, padding: { top: 10, bottom: 10 }, renderLineHighlight: "line",
          lineNumbersMinChars: 3,
        }}
      />
    </div>
  );
}
