"""Central token<->ID mapping shared by every tokenizer in this project.

`Vocabulary` is the single source of truth for how a tokenizer's tokens map
to integer IDs. Character/word/BPE/WordPiece tokenizers register their
tokens here instead of each keeping their own `dict[str, int]`, so the
mapping logic (assignment, duplicate handling, lookup, size) is written and
tested exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable


class Vocabulary:
    """Bidirectional, deterministic mapping between tokens and integer IDs.

    IDs are assigned in registration order starting at 0 and are never
    reused or reassigned: the first token added gets ID 0, the second ID 1,
    and so on. Registering the same token more than once is a no-op that
    returns the token's existing ID — it never creates a duplicate entry or
    changes an ID that was already handed out.
    """

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in tokens or ():
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """Register `token` if new, and return its ID either way.

        Idempotent: adding a token that is already in the vocabulary
        returns its existing ID unchanged, rather than creating a
        duplicate entry.
        """
        existing_id = self._token_to_id.get(token)
        if existing_id is not None:
            return existing_id

        new_id = len(self._token_to_id)
        self._token_to_id[token] = new_id
        self._id_to_token[new_id] = token
        return new_id

    def get_id(self, token: str) -> int:
        """Look up the ID of `token`.

        Raises:
            KeyError: if `token` has never been added to this vocabulary.
        """
        try:
            return self._token_to_id[token]
        except KeyError:
            raise KeyError(f"Unknown token: {token!r}") from None

    def get_token(self, id_: int) -> str:
        """Look up the token registered under `id_`.

        Raises:
            KeyError: if `id_` was never assigned by this vocabulary.
        """
        try:
            return self._id_to_token[id_]
        except KeyError:
            raise KeyError(f"Unknown token id: {id_!r}") from None

    def has_token(self, token: str) -> bool:
        """Return whether `token` is already registered."""
        return token in self._token_to_id

    def has_id(self, id_: int) -> bool:
        """Return whether `id_` has been assigned to a token."""
        return id_ in self._id_to_token

    def __contains__(self, token: str) -> bool:
        return self.has_token(token)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def vocab_size(self) -> int:
        """Total number of distinct tokens registered so far."""
        return len(self._token_to_id)
