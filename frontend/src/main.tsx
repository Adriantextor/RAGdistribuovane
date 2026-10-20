import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Source = { id: string; document: string; page: number; text: string };
type Job = { request_id: string; question: string; status: string; answer: string | null; sources: Source[]; error: string | null };
type Session = { session_id: string; created_at: number };
type Health = { application: boolean; rag: boolean; inference: boolean };

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text.slice(0, 250)}`);
  }
  return response.json();
}

function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState(localStorage.getItem('session') || '');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [question, setQuestion] = useState('');
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState('');
  const [sending, setSending] = useState(false);
  const pending = useRef<{ id: string; question: string; session: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [items, status] = await Promise.all([api<Session[]>('/sessions'), api<Health>('/health')]);
        if (active) { setSessions(items); setHealth(status); }
      } catch { if (active) setHealth(null); }
    };
    void refresh();
    const timer = setInterval(refresh, 5000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    setJobs([]);
    if (!session) return;
    localStorage.setItem('session', session);
    let active = true;
    const refresh = async () => {
      try { const items = await api<Job[]>(`/sessions/${session}/messages`); if (active) setJobs(items); }
      catch (e) { if (active) setError(String(e)); }
    };
    void refresh();
    const timer = setInterval(refresh, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [session]);

  async function newSession() {
    try {
      const item = await api<{ session_id: string }>('/sessions', { method: 'POST' });
      setSession(item.session_id); setError('');
      setSessions(await api<Session[]>('/sessions'));
    } catch (e) { setError(String(e)); }
  }

  async function send(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || !session || sending) return;
    setSending(true); setError('');
    if (!pending.current || pending.current.question !== text || pending.current.session !== session) {
      pending.current = { id: crypto.randomUUID(), question: text, session };
    }
    try {
      const result = await api<Job>(`/sessions/${session}/questions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: pending.current.id, question: text }),
      });
      setJobs(items => items.some(item => item.request_id === result.request_id) ? items : [...items, result]);
      setQuestion(''); pending.current = null;
    } catch (e) { setError(String(e)); }
    finally { setSending(false); }
  }

  async function upload(file?: File) {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) { setError('Maximálna veľkosť PDF je 20 MB.'); return; }
    setUploading(true); setError(''); setUploadResult('');
    const body = new FormData(); body.append('file', file);
    try {
      const result = await api<{ document: string; chunks: number }>('/documents', { method: 'POST', body });
      setUploadResult(`${result.document} · ${result.chunks} chunkov`);
    } catch (e) { setError(String(e)); }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = ''; }
  }

  return <div className="layout">
    <aside>
      <div className="brand"><span className="mark">R</span><div>Research<span>LOCAL AI WORKSPACE</span></div></div>
      <button className="new" onClick={newSession}>＋ Nová konverzácia</button>
      <p className="label">KONVERZÁCIE</p>
      <nav>{sessions.map((item, i) => <button className={session === item.session_id ? 'selected' : ''} key={item.session_id} onClick={() => setSession(item.session_id)}>
        Konverzácia {sessions.length - i}<small>{new Date(item.created_at * 1000).toLocaleString('sk')}</small>
      </button>)}</nav>
      <div className="nodes"><p className="label">DOSTUPNOSŤ UZLOV</p>{(['application', 'rag', 'inference'] as const).map(node =>
        <div key={node}><i className={health?.[node] ? 'online' : ''} />{{ application: 'Aplikácia · Mac', rag: 'RAG · Azure / lokálne', inference: 'Inference · Mac' }[node]}</div>)}</div>
    </aside>
    <main>
      <header><div><span className="eyebrow">DISTRIBUOVANÝ RAG CHATBOT</span><h1>AI Research Assistant</h1></div><span className="badge">Lokálny model</span></header>
      <section className="library"><div><strong>Tvoja knižnica výskumu</strong><p>Začni článkami RAG a QLoRA. Odpovede čerpajú z nahraných PDF.</p></div>
        <input ref={fileInput} type="file" accept="application/pdf" hidden onChange={e => void upload(e.target.files?.[0])} />
        <button disabled={uploading} onClick={() => fileInput.current?.click()}>{uploading ? 'Vytváram index…' : '＋ Nahrať PDF'}</button>
      </section>
      {uploadResult && <p className="notice" role="status">Import dokončený: {uploadResult}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="messages">
        {!jobs.length && <div className="empty"><span className="empty-icon">✳</span><h2>Od otázky k zdroju.</h2><p>Nahraj článok, vytvor konverzáciu a preskúmaj AI.<br />Každú odpoveď môžeš overiť v nájdených úryvkoch.</p><div className="suggestions">{['What is retrieval-augmented generation?', 'How does QLoRA reduce memory usage?'].map(text => <button key={text} onClick={() => setQuestion(text)}>{text} ↗</button>)}</div></div>}
        {jobs.map(job => <article key={job.request_id}>
          <div className="question"><span>TY</span><p>{job.question}</p></div>
          <div className="answer"><span>RESEARCH ASSISTANT</span>{job.answer ? <p>{job.answer}</p> : <p className="waiting">{({ queued: 'Vo fronte', retrieving: 'Vyhľadávam v dokumentoch…', generating: 'Generujem odpoveď…', failed: 'Spracovanie zlyhalo' } as Record<string, string>)[job.status] || job.status}</p>}
            {job.error && !job.answer && <small>{job.error}</small>}
            {job.sources.length > 0 && <div className="sources">{job.sources.map((source, i) => <details key={source.id}><summary>[{i + 1}] {source.document} · strana {source.page}</summary><p>{source.text}</p></details>)}</div>}
          </div>
        </article>)}
      </div>
      <form onSubmit={send}><label className="sr-only" htmlFor="question">Tvoja otázka</label><textarea id="question" maxLength={4000} value={question} onChange={e => setQuestion(e.target.value)} placeholder={session ? 'Spýtaj sa na svoje dokumenty…' : 'Najprv vytvor novú konverzáciu…'} rows={2} /><button disabled={!session || !question.trim() || sending} type="submit">{sending ? '…' : 'Odoslať ↑'}</button></form>
      <footer>Odpovede môžu obsahovať chyby. Over si ich v uvedených zdrojoch.</footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')!).render(<App />);
