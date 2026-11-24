"""A system prompt made an agent's own: a base prompt, its placeholders
filled in with the agent's details.

    prompt, missing = fillPrompt(
        "You are {{persona}}, of {{businessName}}.",
        {"persona": "Alex, a support representative", "businessName": "Acme"},
    )
    # "You are Alex, a support representative, of Acme.", []

A placeholder is a detail's key in double braces - `{{persona}}`, or
`{{ persona }}` - which PLACEHOLDER, one regex, finds, and `re.sub`
replaces with the detail's text, as it is: never read for placeholders
of its own. A placeholder the details have no text for is left empty,
and its key handed back, for the node to log. Single braces, as a JSON
example has, are no placeholder, and kept as they are.

The details are the agent's, as `Agent.detailsOf` hands them to each of
its nodes, in its config's AGENT_DETAILS: its `name` and `description`,
and its record's `metadata` - `persona`, `businessName`, `serviceTone`,
`primaryObjective`, `secondaryObjective`, `marketResearchObjective`.
"""

import re
from collections.abc import Mapping

# The config key the agent hands each node its details in.
AGENT_DETAILS = "agent"
# A placeholder: a detail's key, in double braces.
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")


def fillPrompt(
    prompt: str, details: Mapping[str, str]
) -> tuple[str, list[str]]:
    """`prompt`, each placeholder replaced by the text of its detail; and
    the keys, in order, of those with no text in `details`, left empty."""
    missing = []

    def detailOf(match: re.Match) -> str:
        key = match.group(1)
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if key not in missing:
            missing.append(key)
        return ""

    return PLACEHOLDER.sub(detailOf, prompt), missing
