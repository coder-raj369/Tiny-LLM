"""Small in-memory metrics collector for the API process."""

from collections import Counter
from threading import Lock
from typing import Dict


class APIMetrics:
    """Collect process-local request counters and latency summaries."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._latency_total_ms = 0.0
        self._status_counts: Counter[int] = Counter()
        self._route_counts: Counter[str] = Counter()

    def observe(self, route: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._latency_total_ms += latency_ms
            self._status_counts[status_code] += 1
            self._route_counts[route] += 1

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "request_count": self._request_count,
                "latency_total_ms": self._latency_total_ms,
                "status_counts": dict(self._status_counts),
                "route_counts": dict(self._route_counts),
            }

    def prometheus_text(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP tiny_llm_http_requests_total Total HTTP requests handled.",
            "# TYPE tiny_llm_http_requests_total counter",
            f"tiny_llm_http_requests_total {snapshot['request_count']}",
            "# HELP tiny_llm_http_request_latency_ms_total Cumulative request latency in milliseconds.",
            "# TYPE tiny_llm_http_request_latency_ms_total counter",
            f"tiny_llm_http_request_latency_ms_total {snapshot['latency_total_ms']:.3f}",
        ]
        for status_code, count in sorted(snapshot["status_counts"].items()):
            lines.append(
                f'tiny_llm_http_responses_total{{status_code="{status_code}"}} {count}'
            )
        for route, count in sorted(snapshot["route_counts"].items()):
            escaped_route = route.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'tiny_llm_http_route_requests_total{{route="{escaped_route}"}} {count}')
        return "\n".join(lines) + "\n"
