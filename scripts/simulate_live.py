"""Drive a running API with ordinary traffic followed by deliberate drift."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post(url: str) -> dict:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - user-supplied local URL
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--normal", type=int, default=100)
    parser.add_argument("--suspicious", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()

    for suspicious, count in ((False, args.normal), (True, args.suspicious)):
        for _ in range(count):
            result = post(f"{args.url}/api/simulate?suspicious={str(suspicious).lower()}")
            print(result["transaction_id"], result["decision"], result["fraud_probability"])
            time.sleep(args.delay)

    with urllib.request.urlopen(f"{args.url}/api/drift") as response:  # noqa: S310
        print(json.dumps(json.loads(response.read()), indent=2))


if __name__ == "__main__":
    main()
