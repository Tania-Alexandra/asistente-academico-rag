import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
OBS_DIR = ROOT_DIR / "observability"
METRICS_FILE = OBS_DIR / "metrics_summary.json"
LOG_FILE = OBS_DIR / "agent_events.log"
METRICS_JSONL = OBS_DIR / "metrics.jsonl"


def load_metrics_summary() -> Dict[str, Any]:
    if not METRICS_FILE.exists():
        return {}
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def load_records() -> List[Dict[str, Any]]:
    if not METRICS_JSONL.exists():
        return []
    records = []
    for line in METRICS_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-20:]


def load_logs() -> List[str]:
    if not LOG_FILE.exists():
        return []
    return [line for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()][-20:]


def build_html() -> str:
    summary = load_metrics_summary()
    records = load_records()
    logs = load_logs()
    if not summary:
        summary_message = "Aún no hay métricas disponibles. Ejecuta alguna consulta para generar observabilidad."
    else:
        summary_message = ""

    records_html = "<pre>No hay registros recientes</pre>"
    if records:
        records_html = "<pre>" + html.escape(json.dumps(records, indent=2, ensure_ascii=False)) + "</pre>"

    logs_html = "<pre>No hay logs disponibles</pre>"
    if logs:
        logs_html = "<pre>" + html.escape("\n".join(logs)) + "</pre>"

    return f"""
    <!doctype html>
    <html lang=\"es\">
      <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
        <title>Observabilidad del Asistente RAG</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #112; }}
          .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
          .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
          .card {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
          .card h3 {{ margin: 0 0 8px; font-size: 14px; color: #4b5563; }}
          .card p {{ margin: 0; font-size: 24px; font-weight: 600; }}
          pre {{ white-space: pre-wrap; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; }}
        </style>
      </head>
      <body>
        <div class=\"container\">
          <h1>📈 Dashboard de observabilidad del asistente</h1>
          <p>Monitorea latencia, precisión, consistencia, errores y logs del flujo RAG.</p>
          <p>{summary_message}</p>
          <div class=\"cards\">
            <div class=\"card\"><h3>Consultas</h3><p>{summary.get('request_count', 0)}</p></div>
            <div class=\"card\"><h3>Precisión</h3><p>{summary.get('precision_score', 0.0):.2f}</p></div>
            <div class=\"card\"><h3>Latencia (ms)</h3><p>{summary.get('latency_ms', 0.0):.2f}</p></div>
            <div class=\"card\"><h3>Consistencia</h3><p>{summary.get('consistency_score', 1.0):.2f}</p></div>
            <div class=\"card\"><h3>Tasa de error</h3><p>{summary.get('error_rate', 0.0):.2f}</p></div>
            <div class=\"card\"><h3>Tasa de éxito</h3><p>{summary.get('success_rate', 0.0):.2f}</p></div>
            <div class=\"card\"><h3>Tokens promedio</h3><p>{summary.get('avg_tokens', 0):.0f}</p></div>
            <div class=\"card\"><h3>CPU/RAM promedio</h3><p>{summary.get('avg_cpu_percent', 0.0):.1f}% / {summary.get('avg_memory_mb', 0.0):.1f} MB</p></div>
          </div>
          <h2>Últimas solicitudes</h2>
          {records_html}
          <h2>Logs recientes</h2>
          {logs_html}
        </div>
      </body>
    </html>
    """


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ["/", "/index.html"]:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        body = build_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"📊 Dashboard disponible en http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Deteniendo dashboard...")
        server.server_close()


if __name__ == "__main__":
    main()
