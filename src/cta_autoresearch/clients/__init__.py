"""Client configuration loader for multi-client autoresearch.

Each client is a Python module in this package exporting:
    PRIMARY_REASONS, ACTIONS, REASON_DENYLIST, CONTROL_ACTION_ID,
    PLAN_TIERS, LLM_DOMAIN_CONTEXT, MUTABLE_DIMENSIONS, DIMENSION_CATALOGS,
    ARCHETYPES, row_to_persona(), reason_from_raw()
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType


def load_client(client_id: str | None = None) -> ModuleType:
    """Load a client module by ID.

    Reads CLIENT_ID env var if not provided. Defaults to 'jungle_ai'.
    """
    client_id = client_id or os.environ.get("CLIENT_ID", "jungle_ai")
    return importlib.import_module(f"cta_autoresearch.clients.{client_id}")
