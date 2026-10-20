export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    let detail = 'Služba nie je dostupná.';
    try { const body = await response.json(); detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail); }
    catch { /* A gateway may return HTML instead of JSON. */ }
    throw new Error(`${detail} (${response.status})`);
  }
  return response.json();
}
export const statusText: Record<string, string> = {
  queued: 'Vo fronte', retrieving: 'Vyhľadávam zdroje', generating: 'Generujem odpoveď',
  completed: 'Dokončené', failed: 'Zlyhalo', processing: 'Vytváram index', needs_reindex: 'Vyžaduje nový import',
};
export const duration = (seconds: number | null | undefined) => seconds == null ? '—' : seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(1)} s`;
