Observability and monitoring
Every production CRAG deployment needs full traceability:
Langfuse — trace each request end-to-end: retrieval latency, grader scores per chunk, which branch was taken, final token count and cost.
OpenTelemetry — instrument the FastAPI app; export to your chosen backend (GCP Cloud Trace or Grafana Tempo).
Postgres table: crag_runs — log (run_id, query, decision, chunks_retrieved, chunks_graded_correct, web_search_triggered, latency_ms, total_tokens) for offline analysis.
Alerting — alert when web_search_triggered / total_queries > 0.4 (your internal KB may be stale), and when p95_latency > 8s