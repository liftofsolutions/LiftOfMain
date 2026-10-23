"""A credential, kept from ever being shown.

    credential = Credential(
        {"phone_number_id": "111222333", "access_token": decryptedToken}
    )
    headers = {"Authorization": f"Bearer {credential.reveal('access_token')}"}

Build one from a credential's values the moment they are decrypted, and
pass it on as it is: a node reads a value with `reveal`, at the moment a
call needs it. Nothing else gives a value back. Its repr and str are a
fixed `Credential(<protected>)`, so no log line, print, f-string,
traceback or failed assertion can show what it holds. It has no
`__dict__` for `vars` to dump, JSON cannot encode it, and pickling it -
which would write its values out in the clear - raises. Copying it, with
`copy` or `deepcopy`, gives another protected credential, and `clear`
deletes its values in place.

This keeps a credential from being shown by accident. Python cannot hide
a value from code that goes looking for it, so keep a revealed value out
of variables, attributes and messages, and hand it straight to the call
that needs it.
"""

from typing import NoReturn


class Credential:
    """A credential's values, which only `reveal` gives back."""

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, str]) -> None:
        """Hold its own copy of `values`."""
        self._values = dict(values)

    def reveal(self, key: str) -> str:
        """The value under `key`.

        Raises:
            KeyError: If the credential holds no such value. The error
                names the key, never a value.
        """
        return self._values[key]

    def copy(self) -> "Credential":
        """Another credential, holding the same values."""
        return Credential(self._values)

    def clear(self) -> None:
        """Delete every value it holds."""
        self._values.clear()

    def __bool__(self) -> bool:
        """Whether it holds any value: an empty one counts as none."""
        return bool(self._values)

    def __repr__(self) -> str:
        """The same for every credential, showing nothing it holds."""
        return "Credential(<protected>)"

    __str__ = __repr__

    def __copy__(self) -> "Credential":
        """Another protected credential, for `copy.copy`."""
        return self.copy()

    def __deepcopy__(self, memo: dict) -> "Credential":
        """Another protected credential, for `copy.deepcopy`."""
        return self.copy()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Refuse to be pickled: a pickle would hold the values in the
        clear."""
        raise TypeError("A credential cannot be pickled.")
