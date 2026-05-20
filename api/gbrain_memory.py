"""Vercel serverless: visible/editable GBrain memory API."""
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
from cta_autoresearch.gbrain_memory import (
    archive_memory,
    normalize_memory_list,
    seed_memories,
    summarize_memory,
    upsert_memory,
)


def _client_from_path(path: str) -> str:
    qs = parse_qs(urlparse(path).query)
    return str(qs.get("client", ["jungle_ai"])[0] or "jungle_ai")


def _load_items(client_id: str) -> list[dict]:
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
            items = _load_items(client_id)
            payload = summarize_memory(items, client_id=client_id)
            payload["editable"] = redis_state.available()
            return self._respond(200, payload)
        except Exception as exc:
            return self._respond(500, {"error": str(exc), "type": type(exc).__name__})

    def do_POST(self):
        try:
            if not redis_state.available():
                return self._respond(503, {"error": "Redis not configured; GBrain memory is read-only"})

            raw_length = self.headers.get("Content-Length", "0")
            body = self.rfile.read(int(raw_length or "0"))
            payload = json.loads(body.decode() or "{}")
            if not isinstance(payload, dict):
                return self._respond(422, {"error": "request body must be an object"})

            memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else payload
            client_id = str(
                payload.get("client_id")
                or memory.get("client_id")
                or parse_qs(urlparse(self.path).query).get("client", ["jungle_ai"])[0]
                or "jungle_ai"
            )
            operation = str(payload.get("operation") or payload.get("action") or "upsert").strip().lower()

            items = _load_items(client_id)
            if operation == "archive":
                memory_id = str(payload.get("id") or memory.get("id") or "").strip()
                if not memory_id:
                    return self._respond(422, {"error": "id is required to archive a memory"})
                items = archive_memory(items, memory_id, client_id=client_id)
            else:
                items = upsert_memory(items, memory, client_id=client_id)

            redis_state.set_gbrain_memory(client_id, items)
            summary = summarize_memory(items, client_id=client_id)
            summary["status"] = "saved"
            summary["editable"] = True
            return self._respond(200, summary)
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
