"""Vercel serverless: build Jungle experiment UI spec JSON."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cta_autoresearch import redis_state
from cta_autoresearch.gbrain_memory import normalize_memory_list, seed_memories
from cta_autoresearch.jungle_experiment_spec import build_jungle_experiment_spec


def _client_from_path(path: str) -> str:
    qs = parse_qs(urlparse(path).query)
    return str(qs.get("client", ["jungle_ai"])[0] or "jungle_ai")


def _load_memory(client_id: str) -> list[dict]:
    if not redis_state.available():
        return seed_memories(client_id)
    stored = redis_state.get_gbrain_memory(client_id)
    return normalize_memory_list(stored, client_id=client_id)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._respond(204, {})

    def do_GET(self):
        try:
            client_id = _client_from_path(self.path)
            spec = build_jungle_experiment_spec({"client_id": client_id}, memory_items=_load_memory(client_id))
            return self._respond(200, spec)
        except Exception as exc:
            return self._respond(500, {"error": str(exc), "type": type(exc).__name__})

    def do_POST(self):
        try:
            raw_length = self.headers.get("Content-Length", "0")
            body = self.rfile.read(int(raw_length or "0"))
            payload = json.loads(body.decode() or "{}")
            if not isinstance(payload, dict):
                return self._respond(422, {"error": "request body must be an object"})
            client_id = str(payload.get("client_id") or _client_from_path(self.path))
            spec = build_jungle_experiment_spec(payload, memory_items=_load_memory(client_id))
            return self._respond(200, spec)
        except json.JSONDecodeError:
            return self._respond(400, {"error": "invalid JSON body"})
        except Exception as exc:
            return self._respond(500, {"error": str(exc), "type": type(exc).__name__})

    def _respond(self, status: int, payload: dict) -> None:
        body = b"" if status == 204 else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)
