"""Short secrets that masking makes longer: the rule #99 adds to the masker, as a stand-in masker, and
the secret-named lines that rule masks. A request that fits before masking can then be over its box."""

from __future__ import annotations

import re

from jev_navigator.judgments.secrets import MASK


class ShortSecretMasker:
    """The quoted value of every assignment to a name holding PASSWORD is masked however short it is,
    so a value shorter than the 8-character mask makes the request longer."""

    _VALUE = re.compile(r'PASSWORD\w* = "([^"]+)"')

    def mask(self, text: str, path: str | None = None) -> str:
        return self._VALUE.sub(lambda match: match[0].replace(match[1], MASK), text)

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [match[1] for match in self._VALUE.finditer(text)]


def numbered_secret(line: int) -> str:
    """A 6-character value under a secret-named key, 2 characters longer once masked."""
    return f'    DB_PASSWORD_{line:05d} = "k{line:05d}"\n'


def hunter2(_line: int) -> str:
    """jvn-verifier's measured case: 7 characters under a suffixed secret key, 1 longer once masked."""
    return '    DB_PASSWORD_PROD = "hunter2"\n'
