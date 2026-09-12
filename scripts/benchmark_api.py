"""Measure API latency against a running Tiny LLM server."""

import argparse
import statistics
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1/generate")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")

    latencies_ms = []
    for index in range(args.requests):
        start = time.perf_counter()
        response = requests.post(
            args.url,
            json={"prompt": "What is a qubit?", "max_tokens": 32, "temperature": 0.8},
            timeout=args.timeout,
            headers={"X-Request-ID": f"benchmark-{index}"},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies_ms.append(elapsed_ms)

    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    total_seconds = sum(latencies_ms) / 1000
    print(f"requests={len(latencies_ms)}")
    print(f"mean_ms={statistics.mean(latencies_ms):.2f}")
    print(f"p50_ms={statistics.median(latencies_ms):.2f}")
    print(f"p95_ms={ordered[p95_index]:.2f}")
    print(f"throughput_rps={len(latencies_ms) / total_seconds:.2f}")


if __name__ == "__main__":
    main()
