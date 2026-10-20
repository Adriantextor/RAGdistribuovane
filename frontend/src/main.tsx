import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, duration, statusText } from './api';
import type { Document, Health, Job, Metrics, Session } from './types';
import './style.css';

type View = 'chat' | 'documents' | 'system';
const examples = ['Čo znamená double quantization v QLoRA?', 'Prečo Transformer používa pozičné kódovanie?', 'Ako funguje RAG?'];

function App() {
  const [view, setView] = useState<View>('chat');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState(localStorage.getItem('session') || '');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [language, setLanguage] = useState('auto');
  const [question, setQuestion] = useState('');
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [live, setLive] = useState(false);
  const pending = useRef<{ id: string; signature: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const sessionRef = useRef(session);
  sessionRef.current = session;

  function upsert(job: Job) {
    if (job.session_id !== sessionRef.current) return;
    setJobs(items => {
      const found = items.some(item => item.request_id === job.request_id);
      return (found ? items.map(item => item.request_id === job.request_id ? job : item) : [...items, job])
        .sort((a, b) => a.created_at - b.created_at || a.request_id.localeCompare(b.request_id));
    });
  }

  async function refreshDocuments() {
    const result = await api<{ documents: Document[] }>('/documents');
    setDocuments(result.documents);
    setSelected(ids => ids.filter(id => result.documents.some(doc => doc.id === id && doc.status === 'completed')));
  }

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const results = await Promise.allSettled([api<Session[]>('/sessions'), api<Health>('/health')]);
      if (!active) return;
      if (results[0].status === 'fulfilled') setSessions(results[0].value);
      if (results[1].status === 'fulfilled') setHealth(results[1].value); else setHealth(null);
      try { const data = await api<{ documents: Document[] }>('/documents'); if (active) setDocuments(data.documents); }
      catch { /* RAG being offline must not hide the chat or its durable history. */ }
    };
    void refresh();
    const timer = setInterval(refresh, 4000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (view !== 'system') return;
    let active = true;
    const refresh = async () => { try { const data = await api<Metrics>('/metrics'); if (active) setMetrics(data); } catch { /* Health indicator explains downtime. */ } };
    void refresh();
    const timer = setInterval(refresh, 2000);
    return () => { active = false; clearInterval(timer); };
  }, [view]);

  useEffect(() => {
    setJobs([]); setLive(false);
    if (!session) return;
    localStorage.setItem('session', session);
    let active = true;
    const stream = new EventSource(`/api/sessions/${session}/events`);
    stream.onopen = () => { if (active) setLive(true); };
    stream.onerror = () => { if (active) setLive(false); };
    stream.addEventListener('snapshot', event => { if (active) setJobs(JSON.parse((event as MessageEvent).data)); });
    stream.addEventListener('job', event => { if (active) upsert(JSON.parse((event as MessageEvent).data)); });
    const refresh = async () => {
      try { const items = await api<Job[]>(`/sessions/${session}/messages`); if (active) setJobs(items); }
      catch (e) { if (active) setError(String(e)); }
    };
    void refresh();
    // Recovery fallback for proxies/browsers that interrupt EventSource.
    const timer = setInterval(refresh, 8000);
    return () => { active = false; stream.close(); clearInterval(timer); };
  }, [session]);

  async function newSession() {
    try {
      const item = await api<{ session_id: string }>('/sessions', { method: 'POST' });
      setSession(item.session_id); setView('chat'); setError(''); pending.current = null;
      setSessions(await api<Session[]>('/sessions'));
    } catch (e) { setError(String(e)); }
  }

  async function send(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || !session || sending) return;
    setSending(true); setError('');
    const signature = JSON.stringify([session, text, [...selected].sort(), language]);
    if (pending.current?.signature !== signature) pending.current = { id: crypto.randomUUID(), signature };
    try {
      const result = await api<Job>(`/sessions/${session}/questions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: pending.current!.id, question: text, document_ids: selected, language }),
      });
      upsert(result); setQuestion(''); pending.current = null;
    } catch (e) { setError(String(e)); }
    finally { setSending(false); }
  }

  async function upload(files: File[]) {
    if (!files.length) return;
    setUploading(true); setError(''); setNotice('');
    try {
      for (const file of files) {
        if (file.size > 20 * 1024 * 1024) throw new Error(`${file.name}: maximálna veľkosť je 20 MB.`);
        const body = new FormData(); body.append('file', file);
        await api('/documents', { method: 'POST', body });
        await refreshDocuments();
      }
      setNotice('Dokumenty boli prijaté. Priebeh indexácie nájdeš v knižnici.'); setView('documents');
    } catch (e) { setError(String(e)); }
    finally { setUploading(false); if (fileInput.current) fileInput.current.value = ''; }
  }

  async function retryJob(id: string) {
    try { upsert(await api<Job>(`/requests/${id}/retry`, { method: 'POST' })); setError(''); }
    catch (e) { setError(String(e)); }
  }

  async function documentAction(doc: Document, retry: boolean) {
    try {
      await api(`/documents/${doc.id}${retry ? '/retry' : ''}`, { method: retry ? 'POST' : 'DELETE' });
      await refreshDocuments();
      if (!retry) setNotice('Dokument bol odobratý z vyhľadávania. Opätovným nahratím ho môžeš obnoviť.');
    } catch (e) { setError(String(e)); }
  }

  const ready = documents.filter(doc => doc.status === 'completed');
  const busy = documents.filter(doc => ['queued', 'processing'].includes(doc.status));
  const activeQuestions = jobs.filter(job => ['queued', 'retrieving', 'generating'].includes(job.status));

  return <div className="layout">
    <aside>
      <div className="brand"><span className="mark">R</span><div>Research<span>LOCAL AI WORKSPACE</span></div></div>
      <button className="new" onClick={newSession}>＋ Nová konverzácia</button>
      <div className="side-tools"><button className={view === 'documents' ? 'selected' : ''} onClick={() => setView('documents')}>▤ Dokumenty <b>{ready.length}</b></button><button className={view === 'system' ? 'selected' : ''} onClick={() => setView('system')}>◉ Systém a metriky</button></div>
      <p className="label">KONVERZÁCIE</p>
      <nav>{sessions.map((item, i) => <button className={session === item.session_id && view === 'chat' ? 'selected' : ''} key={item.session_id} onClick={() => { setSession(item.session_id); setView('chat'); setError(''); }}>
        <span className="session-title">{item.title || `Konverzácia ${sessions.length - i}`}</span><small>{new Date(item.created_at * 1000).toLocaleString('sk')}</small>
      </button>)}</nav>
      <div className="nodes"><p className="label">DOSTUPNOSŤ UZLOV</p>{(['application', 'rag', 'inference'] as const).map(node =>
        <div key={node}><i className={health?.[node] ? 'online' : ''} />{{ application: 'Aplikačný uzol', rag: 'RAG uzol', inference: 'AI inference uzol' }[node]}<span>{health?.[node] ? 'Online' : 'Offline'}</span></div>)}</div>
    </aside>
    <main>
      <header><div><span className="eyebrow">DISTRIBUOVANÝ RAG CHATBOT</span><h1>{view === 'chat' ? 'AI Research Assistant' : view === 'documents' ? 'Knižnica dokumentov' : 'Systém a metriky'}</h1></div><span className="badge">{health?.details?.inference?.models?.[0] || 'Lokálny model'}</span></header>
      <input ref={fileInput} type="file" accept="application/pdf" multiple hidden onChange={e => void upload(Array.from(e.target.files || []))} />
      {error && <div className="error" role="alert">{error}<button aria-label="Zavrieť chybu" onClick={() => setError('')}>×</button></div>}
      {notice && <div className="notice" role="status">{notice}<button aria-label="Zavrieť oznámenie" onClick={() => setNotice('')}>×</button></div>}
      {health && (!health.rag || !health.inference) && <div className="offline-banner">Niektorý uzol je offline. Nové otázky sa bezpečne uložia do frontu a po obnovení spojenia sa spracujú.</div>}
      {view === 'documents' && <>
        <section className="library"><div><strong>Vlastné zdroje pre tvoje otázky</strong><p>Textové PDF do 20 MB a 200 strán. Import pokračuje aj po zatvorení webu.</p></div><button disabled={uploading || !health?.rag} onClick={() => fileInput.current?.click()}>{uploading ? 'Nahrávam…' : '＋ Nahrať PDF'}</button></section>
        <div className="document-list">{documents.length === 0 && <div className="empty"><h2>Pridaj prvý dokument.</h2><p>Po indexácii sa jeho obsah stane dostupným pre chat.</p></div>}
          {documents.map(doc => <article className="document-card" key={doc.id}><div className="document-top"><div><h3>{doc.filename}</h3><p>{doc.pages || '—'} strán · {doc.chunks || '—'} chunkov</p></div><span className={`state ${doc.status}`}>{statusText[doc.status] || doc.status}</span></div>
            {['queued', 'processing'].includes(doc.status) && <><progress max={doc.chunks || 1} value={doc.processed} /><p className="muted">{doc.status === 'queued' ? 'Čaká na dostupné uzly' : `${doc.processed} / ${doc.chunks} chunkov`} · pokus {doc.attempts}</p></>}
            {doc.error && <p className="document-error">{doc.error}</p>}
            <div className="document-actions">{doc.status === 'failed' && doc.file_available && <button onClick={() => void documentAction(doc, true)}>Zopakovať import</button>}{!['queued', 'processing'].includes(doc.status) && <button onClick={() => void documentAction(doc, false)}>Odobrať z vyhľadávania</button>}<code>{doc.id.slice(0, 12)}</code></div>
          </article>)}
        </div>
      </>}
      {view === 'system' && <>
        <p className="section-intro">Živý stav frontu, odozvy a cache. Metriky zahŕňajú všetky konverzácie tejto aplikácie.</p>
        <div className="metric-grid">{[
          ['Vo fronte', metrics?.queue_length ?? '—'], ['Spracúvajú sa', metrics?.active ?? '—'],
          ['Dokončené', metrics?.completed ?? '—'], ['Zlyhané', metrics?.failed ?? '—'],
          ['Priemerná odozva', duration(metrics?.mean_response_seconds)], ['95. percentil', duration(metrics?.p95_response_seconds)],
          ['Priemerné čakanie', duration(metrics?.mean_queue_seconds)], ['Zásahy cache', metrics ? `${metrics.cache_hits} · ${(metrics.cache_hit_ratio * 100).toFixed(0)} %` : '—'],
        ].map(([label, value]) => <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
        <div className="system-note"><strong>Riadené spracovanie</strong><p>{health?.worker_concurrency || '—'} aplikačné workery · {health?.details?.inference?.active ?? '—'} aktívnych modelových operácií · {health?.details?.inference?.waiting ?? '—'} čakajúcich na model · {busy.length} importov</p><p>Cache: {metrics?.cache_entries ?? 0} platných položiek. Odozva zahŕňa čakanie vo fronte aj spracovanie; percentily používajú posledných 1000 dokončených otázok.</p></div>
        <h2 className="small-heading">Posledné požiadavky</h2><div className="table-wrap"><table><thead><tr><th>Otázka / ID</th><th>Stav</th><th>Pokusy</th><th>Čas</th><th>Cache</th></tr></thead><tbody>{metrics?.recent.map(job => <tr key={job.request_id}><td><button className="table-question" onClick={() => { setSession(job.session_id); setView('chat'); }}>{job.question.slice(0, 75)}</button><code>{job.request_id.slice(0, 8)}</code></td><td><span className={`state ${job.status}`}>{statusText[job.status]}</span></td><td>{job.attempts}</td><td>{duration(job.total_ms / 1000)}</td><td>{job.cache_hit ? 'Áno' : '—'}</td></tr>)}</tbody></table></div>
      </>}
      {view === 'chat' && <>
        <section className="library"><div><strong>{ready.length} dokumentov pripravených na otázky</strong><p>{busy.length ? `${busy.length} dokumentov sa ešte spracúva.` : 'Odpovede vychádzajú z nahraných zdrojov.'}</p></div><button disabled={uploading || !health?.rag} onClick={() => fileInput.current?.click()}>{uploading ? 'Nahrávam…' : '＋ Nahrať PDF'}</button></section>
        <div className="chat-options"><label>Zdroje<select aria-label="Výber dokumentu" value={selected[0] || ''} onChange={e => setSelected(e.target.value ? [e.target.value] : [])}><option value="">Všetky dokumenty</option>{ready.map(doc => <option key={doc.id} value={doc.id}>{doc.filename}</option>)}</select></label><label>Jazyk<select aria-label="Jazyk odpovede" value={language} onChange={e => setLanguage(e.target.value)}><option value="auto">Podľa otázky</option><option value="sk">Slovenčina</option><option value="en">English</option></select></label><span className={`live ${live ? 'connected' : ''}`}>{live ? '● Živé odpovede' : '○ Obnovujem spojenie'}</span></div>
        <div className="messages">
          {!jobs.length && <div className="empty"><span className="empty-icon">✳</span><h2>Od otázky k zdroju.</h2><p>Vyber dokumenty a preskúmaj ich obsah.<br />Odpoveď môžeš overiť v nájdených úryvkoch.</p><div className="suggestions">{examples.map(text => <button key={text} onClick={() => setQuestion(text)}>{text} ↗</button>)}</div>{!session && <button className="start" onClick={newSession}>Vytvoriť prvú konverzáciu</button>}</div>}
          {jobs.map(job => <article key={job.request_id} className="chat-turn">
            <div className="question"><span>TY</span><p>{job.question}</p></div>
            <div className="answer"><div className="answer-label"><span>RESEARCH ASSISTANT</span><span className={`state ${job.status}`}>{statusText[job.status]}</span>{job.cache_hit && <span className="cache-badge">Z cache</span>}</div>
              {job.answer || job.partial_answer ? <div className={`markdown ${job.status === 'generating' ? 'streaming' : ''}`}><ReactMarkdown remarkPlugins={[remarkGfm]}>{job.answer || job.partial_answer}</ReactMarkdown></div> : <p className="waiting">{job.status === 'queued' ? 'Otázka je bezpečne uložená a čaká na spracovanie.' : job.status === 'retrieving' ? 'Hľadám relevantné časti dokumentov…' : job.status === 'failed' ? 'Otázku sa nepodarilo dokončiť.' : 'Model pripravuje odpoveď…'}</p>}
              {job.error && !job.answer && <p className="job-error">{job.error}</p>}{job.status === 'failed' && <button className="retry" onClick={() => void retryJob(job.request_id)}>Zopakovať otázku</button>}
              {job.sources.length > 0 && <div className="sources">{job.sources.map((source, i) => <details key={source.id}><summary>[{i + 1}] {source.document} · strana {source.page}</summary><p>{source.text}</p><code>Chunk {source.id.slice(0, 8)}</code></details>)}</div>}
              <details className="request-details"><summary>Podrobnosti požiadavky</summary><div><code>{job.request_id}</code><p>Pokusy: {job.attempts} · Čakanie: {duration(job.queue_ms / 1000)} · Vyhľadávanie: {duration(job.retrieval_ms / 1000)} · Generovanie: {duration(job.generation_ms / 1000)} · Celkom: {duration(job.total_ms / 1000)}</p>{job.model && <p>Model: {job.model} · Tokeny: {job.output_tokens} · Prvý token: {duration(job.ttft_ms == null ? null : job.ttft_ms / 1000)}</p>}</div></details>
            </div>
          </article>)}
        </div>
        {activeQuestions.length > 0 && <p className="queue-note">{activeQuestions.length} otázok v tejto konverzácii čaká alebo sa spracúva. Ďalšie otázky zachovajú svoje poradie.</p>}
        <form onSubmit={send}><label className="sr-only" htmlFor="question">Tvoja otázka</label><textarea id="question" maxLength={4000} value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} placeholder={session ? 'Spýtaj sa na svoje dokumenty…' : 'Najprv vytvor novú konverzáciu…'} rows={2} /><button disabled={!session || !question.trim() || sending} type="submit">{sending ? '…' : 'Odoslať ↑'}</button></form>
        <footer>Enter odošle otázku · Shift + Enter vloží riadok · Over si odpoveď v zdrojoch.</footer>
      </>}
    </main>
  </div>;
}

createRoot(document.getElementById('root')!).render(<App />);
