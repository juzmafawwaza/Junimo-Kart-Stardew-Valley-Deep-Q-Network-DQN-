from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from junimo_rl import JunimoKartBridgeClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Junimo Kart RL bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start", action="store_true", help="Request a fresh Progress Mode run.")
    parser.add_argument("--mode", default="progress", choices=["progress", "endless", "infinite"])
    parser.add_argument("--hold", type=float, default=0.0, help="Hold jump for N seconds as a tiny input test.")
    args = parser.parse_args()

    with JunimoKartBridgeClient(host=args.host, port=args.port) as client:
        print(json.dumps(client.ping(), indent=2))
        if args.start:
            print(json.dumps(client.start(args.mode), indent=2))
            time.sleep(0.25)

        if args.hold > 0:
            client.action(True)
            time.sleep(args.hold / 2)
            held_state = client.state()
            print(json.dumps({"duringHold": held_state}, indent=2))
            time.sleep(args.hold / 2)
            client.action(False)

        state = client.state()
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
