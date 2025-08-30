
import React, { useCallback, useMemo, useRef, useState } from 'react';
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
      className="animate-spin h-4 w-4 mr-2 text-white inline"
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
    <div className="min-h-screen bg-white flex flex-col">
      <header className="bg-white text-gray-900 font-bold text-center p-4 shadow-sm">
        RAG from PDFs
      </header>
      <main className="flex-grow container mx-auto p-4 flex-col">
        <div className="flex-grow bg-white shadow overflow-hidden sm:rounded-lg my-4">
          <div className="p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-gray-500 text-sm">Haz una pregunta sobre los PDFs cargados.</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`p-3 rounded-lg ${m.role === 'user' ? 'bg-blue-50 text-gray-800' : 'bg-gray-100 text-gray-800'}`}>
                <div className="text-xs mb-1 font-semibold uppercase tracking-wide text-gray-500">{m.role}</div>
                <div className="whitespace-pre-wrap">{m.content}</div>
                {/* Sources ocultos en la UI por petición */}
              </div>
            ))}
          </div>
          <div className="p-4 bg-gray-100">
            {error && <div className="text-red-600 text-sm mb-2">{error}</div>}
            {ingestInfo && <div className="text-green-700 text-sm mb-2">{ingestInfo}</div>}
            <textarea
              className="form-textarea w-full p-2 border rounded text-gray-700 bg-white border-gray-300 resize-none h-auto"
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
                className="bg-blue-500 hover:bg-blue-600 disabled:opacity-50 text-white font-bold py-2 px-4 rounded transition duration-150 ease-in-out"
              >
                {loading ? 'Enviando…' : 'Enviar'}
              </button>
              {loading && (
                <button
                  onClick={() => abortRef.current?.abort()}
                  className="text-gray-600 hover:text-gray-800 text-sm"
                >
                  Cancelar
                </button>
              )}
            </div>
            <div className="p-4 bg-gray-50 mt-4 rounded">
              <div className="text-sm font-semibold mb-2">Subir PDFs</div>
              <input
                type="file"
                accept=".pdf"
                multiple
                onChange={(e) => setSelectedFiles(e.target.files)}
                disabled={uploading}
              />
              <button
                className="ml-2 mt-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
                onClick={handleUploadFiles}
                disabled={uploading || !selectedFiles || selectedFiles.length === 0}
              >
                {uploading ? (<><Spinner />Subiendo…</>) : 'Upload PDFs'}
              </button>
              <div className="mt-4">
                <div className="text-sm font-semibold mb-2">Reindexar</div>
                <button
                  className="bg-gray-700 hover:bg-gray-800 text-white font-bold py-1 px-3 rounded disabled:opacity-50 mr-2"
                  onClick={() => handleReindex('update')}
                  disabled={reindexing}
                >
                  {reindexing && reindexMode === 'update' ? (<><Spinner />Reindexando…</>) : 'Reindexar (update)'}
                </button>
                <button
                  className="bg-gray-500 hover:bg-gray-600 text-white font-bold py-1 px-3 rounded disabled:opacity-50"
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
      <footer className="bg-white text-gray-700 text-center p-4 text-xs border-t border-gray-200">
        Backend: {API_BASE} | PDFs: {backendStaticBase}
      </footer>
    </div>
  );
}

export default App;

