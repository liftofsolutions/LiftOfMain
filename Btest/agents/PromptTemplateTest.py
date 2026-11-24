"""Btest for `agents.nodes.promptTemplate`: a base prompt, its
placeholders filled in with an agent's details.

1. Each `{{key}}` - space inside the braces or not - is replaced by the
   text of its detail, however often it comes.
2. A placeholder with no text in the details - none, blank, or not text -
   is left empty, and its key handed back, once, in order.
3. A detail's text is put in as it is: never read for placeholders of
   its own. Single braces, as a JSON example has, are kept.

    uv run pytest Btest/agents -q
"""

import pytest

from agents.nodes.promptTemplate import fillPrompt

DETAILS = {
    "name": "Acme Sales Bot",
    "persona": "Alex, a customer support representative",
    "businessName": "Acme",
    "serviceTone": "Polite and very courteous",
    "primaryObjective": "Help customers resolve their needs",
    "secondaryObjective": "Mention Product 1 when it is relevant",
    "marketResearchObjective": "Learn which products customers prefer",
}


class TestFillPrompt:
    def testEachPlaceholderIsReplacedByItsDetail(self):
        prompt = "\n".join(f"{key}: {{{{{key}}}}}" for key in DETAILS)

        filled, missing = fillPrompt(prompt, DETAILS)

        assert filled == "\n".join(
            f"{key}: {value}" for key, value in DETAILS.items()
        )
        assert missing == []

    def testSpaceInsideTheBracesAndRepeatsAreFilledToo(self):
        filled, _ = fillPrompt(
            "{{ businessName }} - {{businessName}}", DETAILS
        )

        assert filled == "Acme - Acme"

    @pytest.mark.parametrize(
        "details",
        [{}, {"persona": ""}, {"persona": "   "}, {"persona": 7}],
        ids=["none", "empty", "blank", "notText"],
    )
    def testADetailWithNoTextIsLeftEmptyAndNamed(self, details):
        filled, missing = fillPrompt(
            "You are {{persona}}. Be {{persona}} for {{businessName}}.",
            details,
        )

        assert filled == "You are . Be  for ."
        assert missing == ["persona", "businessName"]

    def testADetailsTextIsNotReadForPlaceholders(self):
        details = {"persona": "{{businessName}}", "businessName": "Acme"}

        filled, missing = fillPrompt("You are {{persona}}.", details)

        assert filled == "You are {{businessName}}."
        assert missing == []

    def testSingleBracesAndAPromptWithNoPlaceholdersAreKept(self):
        prompt = 'Answer as {"message": "...", "next": "output"}; be {kind}.'

        assert fillPrompt(prompt, DETAILS) == (prompt, [])
