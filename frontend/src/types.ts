export type Source = { id: string; document_id: string; document: string; page: number; text: string; score: number };
export type Job = {
  request_id: string; session_id: string; question: string; status: string; answer: string | null;
  partial_answer: string; sources: Source[]; attempts: number; error: string | null; cache_hit: boolean;
  created_at: number; completed_at: number | null; total_ms: number; queue_ms: number;
  retrieval_ms: number; generation_ms: number; ttft_ms: number | null; model: string; output_tokens: number;
};
export type Session = { session_id: string; created_at: number; title: string; message_count: number };
export type Document = { id: string; filename: string; model: string; status: string; chunks: number;
  processed: number; pages: number; error: string | null; file_available: boolean; attempts: number };
export type Health = { application: boolean; rag: boolean; inference: boolean; worker_concurrency: number;
  details: { inference: { models?: string[]; active?: number; waiting?: number }; rag: { documents?: number } } };
export type Metrics = { queue_length: number; active: number; completed: number; failed: number;
  cache_hits: number; cache_entries: number; cache_hit_ratio: number; mean_response_seconds: number | null;
  p50_response_seconds: number | null; p95_response_seconds: number | null; mean_queue_seconds: number | null;
  oldest_queued_seconds: number; states: Record<string, number>; recent: Job[] };
