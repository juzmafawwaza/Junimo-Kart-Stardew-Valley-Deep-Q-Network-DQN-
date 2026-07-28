from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class JunimoKartBridgeClient:
    host: str = "127.0.0.1"
    port: int = 8765
    timeout: float = 5.0
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _file: Any = field(default=None, init=False, repr=False)

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._file = sock.makefile("rwb", buffering=0)

    def close(self) -> None:
        file_obj = self._file
        sock = self._sock
        self._file = None
        self._sock = None
        if file_obj is not None:
            file_obj.close()
        if sock is not None:
            sock.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._request_once(payload)
        except (ConnectionError, OSError):
            self.close()
            return self._request_once(payload)

    def _request_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self.connect()
            assert self._file is not None
            raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            self._file.write(raw)
            line = self._file.readline()
        except (ConnectionError, OSError) as exc:
            self.close()
            raise ConnectionError(
                f"Cannot reach Junimo Kart bridge at {self.host}:{self.port}. "
                "Pastikan Stardew dibuka lewat SMAPI dan save sudah loaded."
            ) from exc

        if not line:
            self.close()
            raise ConnectionError(
                f"Junimo Kart bridge at {self.host}:{self.port} closed the connection. "
                "Pastikan Stardew/SMAPI masih running."
            )

        response = json.loads(line.decode("utf-8"))
        if not response.get("ok", False):
            raise RuntimeError(response.get("message") or f"Bridge request failed: {response!r}")
        return response

    def ping(self) -> dict[str, Any]:
        return self.request({"type": "ping"})

    def state(self) -> dict[str, Any]:
        return self.request({"type": "state"})["state"]

    def start(self, mode: str = "progress") -> dict[str, Any]:
        return self.request({"type": "start", "mode": mode})

    def advance(self) -> dict[str, Any]:
        return self.request({"type": "advance"})["state"]

    def action(self, jump: bool) -> dict[str, Any]:
        return self.request({"type": "action", "jump": bool(jump)})["state"]

    def __enter__(self) -> "JunimoKartBridgeClient":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
