"""Btest for `agents.nodes.credential`, the credential never shown.

A credential's values come back from `reveal` alone. Everything else
that could show them - repr and str, f-strings, a log line, a failed
assertion, a dict or list holding it, `vars`, JSON and pickle - shows a
fixed text or refuses, and every check looks for the values themselves in
what comes out. Copies are protected credentials too, and `clear`
deletes the values in place.

    uv run pytest Btest/agents -q
"""

import copy
import json
import logging
import pickle

import pytest

from agents.nodes import Credential

PROTECTED = "Credential(<protected>)"


def buildValues(**overrides) -> dict:
    """A WhatsApp credential's values, decrypted."""
    values = {
        "phone_number_id": "111222333",
        "access_token": "acme-secret-token",
    }
    values.update(overrides)
    return values


def buildCredential(**overrides) -> Credential:
    """A WhatsApp credential, protected."""
    return Credential(buildValues(**overrides))


def assertShowsNoValue(text: str) -> None:
    """Fail if `text` holds any of the credential's values."""
    for value in buildValues().values():
        assert value not in text


class TestReveal:
    """The one way a value comes back."""

    def testRevealGivesBackEachValue(self):
        credential = buildCredential()

        assert credential.reveal("access_token") == "acme-secret-token"
        assert credential.reveal("phone_number_id") == "111222333"

    def testMissingKeyNamesTheKeyAndNoValue(self):
        credential = buildCredential()

        with pytest.raises(KeyError) as raised:
            credential.reveal("password")

        assert "password" in str(raised.value)
        assertShowsNoValue(str(raised.value))

    def testCredentialKeepsItsOwnCopyOfTheValues(self):
        values = buildValues()

        credential = Credential(values)
        values["access_token"] = "changed-by-the-caller"

        assert credential.reveal("access_token") == "acme-secret-token"


class TestNeverShown:
    """Nothing but `reveal` shows a value."""

    def testReprAndStrShowNothing(self):
        credential = buildCredential()

        assert repr(credential) == PROTECTED
        assert str(credential) == PROTECTED
        assert f"{credential}" == PROTECTED
        assert f"{credential!r}" == PROTECTED

    def testLogLineShowsNothing(self, caplog):
        caplog.set_level(logging.INFO)

        logging.getLogger(__name__).info(
            "Using %s and %r.", buildCredential(), buildCredential()
        )

        assert caplog.text.count(PROTECTED) == 2
        assertShowsNoValue(caplog.text)

    def testFailedAssertionShowsNothing(self):
        credential = buildCredential()

        with pytest.raises(AssertionError) as raised:
            assert credential is None

        assert PROTECTED in str(raised.value)
        assertShowsNoValue(str(raised.value))

    def testDictOrListHoldingItShowsNothing(self):
        holder = {"credential": buildCredential(), "all": [buildCredential()]}

        assertShowsNoValue(repr(holder))
        assertShowsNoValue(str(holder))

    def testVarsCannotDumpIt(self):
        with pytest.raises(TypeError):
            vars(buildCredential())

    def testJsonCannotEncodeIt(self):
        with pytest.raises(TypeError):
            json.dumps(buildCredential())

        encoded = json.dumps({"credential": buildCredential()}, default=str)
        assertShowsNoValue(encoded)

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def testPickleIsRefused(self, protocol):
        with pytest.raises(TypeError):
            pickle.dumps(buildCredential(), protocol=protocol)


class TestCopiesAndClearing:
    """Copies, and deleting the values."""

    @pytest.mark.parametrize(
        "makeCopy",
        [Credential.copy, copy.copy, copy.deepcopy],
        ids=["copy", "copyModule", "deepcopy"],
    )
    def testCopyIsAnotherProtectedCredential(self, makeCopy):
        credential = buildCredential()

        copied = makeCopy(credential)
        credential.clear()

        assert copied is not credential
        assert copied.reveal("access_token") == "acme-secret-token"
        assert repr(copied) == PROTECTED

    def testClearDeletesEveryValue(self):
        credential = buildCredential()

        credential.clear()

        assert not credential
        with pytest.raises(KeyError):
            credential.reveal("access_token")

    @pytest.mark.parametrize(
        "values, isHeld", [(buildValues(), True), ({}, False)]
    )
    def testOnlyACredentialWithValuesCountsAsOne(self, values, isHeld):
        assert bool(Credential(values)) is isHeld
