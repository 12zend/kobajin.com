"""Load the feed role prompts and sampling profiles shared with the website.

The browser reads src/prompts/roles.json and the role .txt files through
src/agents.js. This module composes system prompts the same way so evaluation
runs use exactly the prompts and decoding settings the site uses:

  append      system_prompt.txt + role prompt (+ context)
  standalone  role prompt (+ context)

Parts are stripped and joined with a blank line, matching composeSystemPrompt().
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO / "src" / "prompts"
ROLES_PATH = PROMPTS_DIR / "roles.json"
SYSTEM_PROMPT_PATH = REPO / "self_twin" / "inference" / "system_prompt.txt"


def load_role_config() -> dict:
    return json.loads(ROLES_PATH.read_text(encoding="utf-8"))


def role_names() -> list[str]:
    return sorted(load_role_config()["roles"])


def role_settings(role: str) -> dict:
    roles = load_role_config()["roles"]
    if role not in roles:
        raise KeyError(f"Unknown role {role!r}; expected one of {sorted(roles)}")
    return roles[role]


def role_prompt_text(role: str) -> str:
    return (PROMPTS_DIR / role_settings(role)["prompt"]).read_text(encoding="utf-8").strip()


def compose_system_prompt(role: str, context: str = "") -> str:
    settings = role_settings(role)
    parts = [role_prompt_text(role)]
    if settings["compose"] == "append":
        parts.insert(0, SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip())
    elif settings["compose"] != "standalone":
        raise ValueError(f"Role {role!r} has unknown compose mode {settings['compose']!r}")
    if context and context.strip():
        parts.append(context.strip())
    return "\n\n".join(parts)


def role_prompt_sha256(role: str) -> str:
    return hashlib.sha256(role_prompt_text(role).encode("utf-8")).hexdigest()
