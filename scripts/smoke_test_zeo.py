"""Smoke test: full Zeo swarm ideation + candidate generation pipeline."""
import json
import os
os.environ.setdefault("CLIENT_ID", "zeo_auto")

from cta_autoresearch.clients import load_client
from cta_autoresearch.strategy_policy import configure_catalogs, all_candidates
from cta_autoresearch.simulator import configure_domain
from cta_autoresearch.user_model import configure_dimensions, configure_archetypes, configure_action_candidates
from cta_autoresearch.research_settings import ResearchSettings, AGENT_ROLES as SWARM_ROLES
from cta_autoresearch.swarm_ideation import generate_ideas

client = load_client("zeo_auto")
configure_catalogs(client.DIMENSION_CATALOGS)
configure_dimensions(client.MUTABLE_DIMENSIONS)
configure_archetypes(client.ARCHETYPES)
configure_action_candidates(client.ACTION_TO_CANDIDATE)
configure_domain(client.LLM_DOMAIN_CONTEXT)

with open("data/zeo_eval_cohort.json") as f:
    data = json.load(f)
rows = data if isinstance(data, list) else data.get("rows", [])
rows = rows[:5]
personas = [client.row_to_persona(r, i) for i, r in enumerate(rows)]

settings = ResearchSettings(
    model_name="gpt-4o",
    ideation_agents=len(SWARM_ROLES),
    idea_proposals_per_agent=1,
    openai_reasoning_effort="medium",
)
candidate_universe = all_candidates(settings)
print(f"Candidates: {len(candidate_universe)}")
print(f"Personas: {len(personas)}")
print(f"Roles: {len(SWARM_ROLES)}")

proposals, warnings = generate_ideas(personas, candidate_universe, settings)
print(f"Proposals: {len(proposals)}")
for w in warnings:
    print(f"Warning: {w}")
for p in proposals[:3]:
    print(f"  [{p.agent_role}] {p.label}")
print("PASS")
