# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

r"""Simple async load generator for a running Pixano Inference server.

Fires a fixed number of requests at a chosen concurrency and reports throughput and latency
percentiles. Point it at a probe route (``/health``, ``/v1/ready``) to load the ingress, or at
an inference route (with a deployed model) to watch Serve autoscale the replica count under
load. Uses only ``httpx`` so it runs from the lightweight client env.

Examples:
    # Hammer the readiness probe (no model needed)
    python scripts/load_test.py --url http://localhost:7463 --path /v1/ready -n 500 -c 50

    # Load a deployed detection model and watch replicas rise (see /v1/metrics, /v1/info)
    python scripts/load_test.py --url http://localhost:7463 \\
        --post /v1/inference/detection \\
        --json '{"model": "det", "image": "https://example.com/cat.jpg", "classes": ["cat"]}' \\
        -n 1000 -c 100 --api-key "$PIXANO_INFERENCE_API_KEYS"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct* percentile of *values* (nearest-rank)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


async def _worker(
    client: httpx.AsyncClient,
    queue: asyncio.Queue[int],
    method: str,
    path: str,
    body: dict | None,
    headers: dict[str, str],
    latencies: list[float],
    statuses: list[int],
) -> None:
    """Pull request indices off the queue and fire them until it drains."""
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        start = time.perf_counter()
        try:
            response = await client.request(method, path, json=body, headers=headers)
            status = response.status_code
        except Exception:
            status = 0
        latencies.append(time.perf_counter() - start)
        statuses.append(status)
        queue.task_done()


async def run(args: argparse.Namespace) -> int:
    """Run the load test described by *args* and print a latency/throughput summary."""
    method = "POST" if args.post else "GET"
    path = args.post or args.path
    body = json.loads(args.json) if args.json else None
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(args.num_requests):
        queue.put_nowait(i)

    latencies: list[float] = []
    statuses: list[int] = []
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), timeout=args.timeout, limits=limits) as client:
        workers = [
            asyncio.create_task(_worker(client, queue, method, path, body, headers, latencies, statuses))
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
    wall = time.perf_counter() - wall_start

    ok = sum(1 for s in statuses if 200 <= s < 300)
    failed = len(statuses) - ok
    print(f"{method} {args.url.rstrip('/')}{path}")
    print(f"  requests:     {len(statuses)}  (concurrency {args.concurrency})")
    print(f"  ok / failed:  {ok} / {failed}")
    print(f"  wall time:    {wall:.2f}s")
    print(f"  throughput:   {len(statuses) / wall:.1f} req/s" if wall > 0 else "  throughput:   n/a")
    print(f"  latency p50:  {_percentile(latencies, 50) * 1000:.1f} ms")
    print(f"  latency p90:  {_percentile(latencies, 90) * 1000:.1f} ms")
    print(f"  latency p99:  {_percentile(latencies, 99) * 1000:.1f} ms")
    if statuses:
        by_status: dict[int, int] = {}
        for s in statuses:
            by_status[s] = by_status.get(s, 0) + 1
        print(f"  status codes: {dict(sorted(by_status.items()))}")
    return 0 if failed == 0 else 1


def main() -> int:
    """Parse args and run the load test."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:7463", help="Server base URL.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--path", default="/v1/ready", help="GET path to hit (default: /v1/ready).")
    target.add_argument("--post", help="POST path to hit (use with --json).")
    parser.add_argument("--json", help="JSON body for --post requests.")
    parser.add_argument("-n", "--num-requests", type=int, default=200, help="Total requests to send.")
    parser.add_argument("-c", "--concurrency", type=int, default=20, help="Concurrent in-flight requests.")
    parser.add_argument("--api-key", help="API key sent as X-API-Key.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (s).")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
