
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import './App.css';

type Source = {
  title?: string;
  page?: number;
  source?: string;
  metadata?: Record<string, any>;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
};

const API_BASE = (process.env.REACT_APP_API_URL || 'http://localhost:8080/rag').replace(/\/$/, '');
const STREAM_URL = `${API_BASE}/stream`;
const INVOKE_URL = `${API_BASE}/invoke`;

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [reindexMode, setReindexMode] = useState<'update' | 'full' | null>(null);
  const [ingestInfo, setIngestInfo] = useState<string | null>(null);
  const [theme, setTheme] = useState<'system' | 'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark' || saved === 'system') return saved;
    return 'system';
  });

  // Tailwind darkMode: 'class' — aplica tema según preferencia o sistema
  useEffect(() => {
    const root = document.documentElement;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      const useDark = theme === 'dark' || (theme === 'system' && mql.matches);
      root.classList.toggle('dark', useDark);
    };
    apply();
    if (theme === 'system') {
      mql.addEventListener('change', apply);
      return () => mql.removeEventListener('change', apply);
    }
  }, [theme]);

  const cycleTheme = () => {
    const next = theme === 'system' ? 'dark' : theme === 'dark' ? 'light' : 'system';
    setTheme(next);
    localStorage.setItem('theme', next);
  };

  const backendStaticBase = useMemo(() => {
    try {
      const u = new URL(API_BASE);
      return `${u.protocol}//${u.host}/rag/static`;
    } catch {
      return 'http://localhost:8080/rag/static';
    }
  }, []);

  const ORIGIN = useMemo(() => {
    try {
      const u = new URL(API_BASE);
      return `${u.protocol}//${u.host}`;
    } catch {
      return 'http://localhost:8080';
    }
  }, []);
  const UPLOAD_URL = `${ORIGIN}/upload`;
  const INGEST_URL = `${ORIGIN}/admin/ingest`;

  const Spinner = () => (
    <svg
      className="motion-safe:animate-spin h-4 w-4 mr-2 text-white inline"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
    </svg>
  );

  const computePdfLink = useCallback(
    (src?: string) => {
      if (!src) return undefined;
      try {
        // Si ya es URL absoluta, devuélvela tal cual
        const u = new URL(src);
        return u.toString();
      } catch {
        // Extraer el nombre del archivo y servir desde /rag/static
        const base = src.split('\\').pop()?.split('/').pop();
        if (!base) return undefined;
        return `${backendStaticBase}/${encodeURIComponent(base)}`;
      }
    },
    [backendStaticBase]
  );

  const handleSend = useCallback(async () => {
    const question = input.trim();
    if (!question || loading) return;
    setError(null);
    setLoading(true);
    setInput('');

    // Añadir mensaje del usuario y placeholder para asistente
    setMessages(prev => [...prev, { role: 'user', content: question }, { role: 'assistant', content: '', sources: [] }]);

    const controller = new AbortController();
    abortRef.current = controller;

    let gotStream = false;
    let accumAnswer = '';
    let finalSources: Source[] | undefined = undefined;

    const updateAssistant = (partial: Partial<Message>) => {
      setMessages(prev => {
        const idx = [...prev].reverse().findIndex(m => m.role === 'assistant');
        if (idx === -1) return prev;
        const rIdx = prev.length - 1 - idx;
        const updated = [...prev];
        updated[rIdx] = { ...updated[rIdx], ...partial } as Message;
        return updated;
      });
    };

    try {
      await fetchEventSource(STREAM_URL, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ input: { question } }),
        onopen: async (res) => {
          if (res.ok) return; // ok
          throw new Error(`Stream open failed: ${res.status} ${res.statusText}`);
        },
        onmessage: (ev) => {
          gotStream = true;
          const { data } = ev;
          if (!data) return;
          // Intentar parsear JSON, si no, tratarlo como token de texto
          try {
            const obj = JSON.parse(data);
            // Heurística: buscar campos típicos
            // 1) Fragmentos de respuesta como texto incremental
            const token = obj.token ?? obj.data ?? obj.chunk ?? obj.answer;
            if (typeof token === 'string') {
              accumAnswer += token;
              updateAssistant({ content: accumAnswer });
            }
            // 2) Fuentes en un campo conocido
            const srcs = obj.sources || (obj.output && obj.output.sources);
            if (Array.isArray(srcs)) {
              finalSources = srcs as Source[];
              updateAssistant({ sources: finalSources });
            }
            // 3) Resultado final con answer completo
            const final = obj.output || obj.final || obj.result;
            if (final && typeof final.answer === 'string') {
              accumAnswer = final.answer;
              finalSources = final.sources;
              updateAssistant({ content: accumAnswer, sources: finalSources });
            }
          } catch {
            // Texto plano
            accumAnswer += data;
            updateAssistant({ content: accumAnswer });
          }
        },
        onerror: (err) => {
          throw err;
        },
        onclose: () => {
          // El servidor cerró el stream
        },
      });

      // Si no hubo stream utilizable, hacemos fallback a invoke
      if (!gotStream || !accumAnswer) {
        const res = await fetch(INVOKE_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: { question } }),
        });
        if (!res.ok) throw new Error(`Invoke failed: ${res.status}`);
        const json = await res.json();
        const out = json?.output ?? json;
        accumAnswer = out?.answer ?? String(out ?? '');
        finalSources = out?.sources;
        updateAssistant({ content: accumAnswer, sources: finalSources });
      }
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error en la comunicación con el backend');
      updateAssistant({ content: accumAnswer || 'Lo siento, hubo un error.' });
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [input, loading]);

  const handleUploadFiles = useCallback(async () => {
    if (!selectedFiles || uploading) return;
    setError(null);
    setUploading(true);
    const formData = new FormData();
    Array.from(selectedFiles).forEach((file) => formData.append('files', file));
    try {
      const response = await fetch(UPLOAD_URL, { method: 'POST', body: formData });
      if (!response.ok) {
        let detail = `Upload failed: ${response.status}`;
        try { const j = await response.json(); if (j?.detail) detail = j.detail; } catch {}
        throw new Error(detail);
      }
      await response.json();
      // Trigger reingest in update mode to avoid duplicates
      const ingest = await fetch(`${INGEST_URL}?mode=update`, { method: 'POST' });
      if (!ingest.ok) {
        let detail = `Ingest failed: ${ingest.status}`;
        try { const j = await ingest.json(); if (j?.detail) detail = j.detail; } catch {}
        throw new Error(detail);
      }
      const data = await ingest.json();
      setIngestInfo(`Reingesta (update) completada: archivos=${data.files}, chunks=${data.added_chunks}, borrados=${data.deleted}`);
      setSelectedFiles(null);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error subiendo los archivos');
    } finally {
      setUploading(false);
    }
  }, [selectedFiles, uploading, UPLOAD_URL, INGEST_URL]);

  const handleReindex = useCallback(async (mode: 'update' | 'full') => {
    if (reindexing) return;
    setError(null);
    setIngestInfo(null);
    setReindexMode(mode);
    setReindexing(true);
    try {
      const res = await fetch(`${INGEST_URL}?mode=${mode}`, { method: 'POST' });
      if (!res.ok) {
        let detail = `Ingest failed: ${res.status}`;
        try { const j = await res.json(); if (j?.detail) detail = j.detail; } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      setIngestInfo(`Reindex (${mode}) ok: archivos=${data.files}, chunks=${data.added_chunks}, borrados=${data.deleted}`);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || 'Error en reindexado');
    } finally {
      setReindexing(false);
      setReindexMode(null);
    }
  }, [reindexing, INGEST_URL]);

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 flex flex-col transition-colors">
      <header className="bg-white/70 dark:bg-neutral-900/70 backdrop-blur text-neutral-900 dark:text-neutral-100 p-4 shadow-sm border-b border-neutral-200/60 dark:border-neutral-800/60 transition-colors">
        <div className="container mx-auto max-w-6xl flex items-center justify-between">
          <div className="font-semibold">RAG from PDFs</div>
          <button
            onClick={cycleTheme}
            className="btn-primary"
            title={`Tema: ${theme}`}
          >
            {theme === 'system' ? 'Sistema' : theme === 'dark' ? 'Oscuro' : 'Claro'}
          </button>
        </div>
      </header>
      <main className="flex-grow container mx-auto p-4 flex-col max-w-6xl">
        <div className="flex-grow card overflow-hidden my-4 transition-colors">
          <div className="p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-gray-500 text-sm">Haz una pregunta sobre los PDFs cargados.</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`p-3 rounded-xl transition-colors ${m.role === 'user' ? 'bg-neutral-200 dark:bg-neutral-700 text-neutral-900 dark:text-neutral-100' : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-100'}`}>
                <div className="text-xs mb-1 font-semibold uppercase tracking-wide text-neutral-500 dark:text-neutral-400">{m.role}</div>
                <div className="whitespace-pre-wrap">{m.content}</div>
                {/* Sources ocultos en la UI por petición */}
              </div>
            ))}
          </div>
          <div className="p-4 bg-neutral-100 dark:bg-neutral-800 transition-colors">
            {error && <div className="text-red-600 dark:text-red-400 text-sm mb-2">{error}</div>}
            {ingestInfo && <div className="text-green-700 dark:text-green-400 text-sm mb-2">{ingestInfo}</div>}
            <textarea
              className="input resize-none h-auto"
              placeholder="Escribe tu pregunta. Shift+Enter para nueva línea, Enter para enviar."
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <div className="flex items-center gap-2 mt-2">
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="btn-primary"
              >
                {loading ? 'Enviando…' : 'Enviar'}
              </button>
              {loading && (
                <button
                  onClick={() => abortRef.current?.abort()}
                  className="btn-ghost text-sm"
                >
                  Cancelar
                </button>
              )}
            </div>
            <div className="p-4 bg-neutral-50 dark:bg-neutral-900 mt-4 rounded transition-colors">
              <div className="text-sm font-semibold mb-2">Subir PDFs</div>
              <input
                type="file"
                accept=".pdf"
                multiple
                onChange={(e) => setSelectedFiles(e.target.files)}
                disabled={uploading}
              />
              <button
                className="mt-2 btn-primary"
                onClick={handleUploadFiles}
                disabled={uploading || !selectedFiles || selectedFiles.length === 0}
              >
                {uploading ? (<><Spinner />Subiendo…</>) : 'Upload PDFs'}
              </button>
              <div className="mt-4">
                <div className="text-sm font-semibold mb-2">Reindexar</div>
                <button
                  className="btn-primary mr-2"
                  onClick={() => handleReindex('update')}
                  disabled={reindexing}
                >
                  {reindexing && reindexMode === 'update' ? (<><Spinner />Reindexando…</>) : 'Reindexar (update)'}
                </button>
                <button
                  className="btn-primary"
                  onClick={() => handleReindex('full')}
                  disabled={reindexing}
                  title="Elimina la colección y reingesta todo"
                >
                  {reindexing && reindexMode === 'full' ? (<><Spinner />Reindexando…</>) : 'Reindexar (full)'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
      <footer className="bg-white/70 dark:bg-neutral-900/70 backdrop-blur text-neutral-700 dark:text-neutral-300 text-center p-4 text-xs border-t border-neutral-200/60 dark:border-neutral-800/60 transition-colors">
        Backend: {API_BASE} | PDFs: {backendStaticBase}
      </footer>
    </div>
  );
}

export default App;

