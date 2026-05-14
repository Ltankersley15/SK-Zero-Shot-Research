from __future__ import annotations


class SearchManager:
    """Small policy helper for candidate promotion during search."""

    @staticmethod
    def allow_candidate(confirm_ok: bool, plausibility_ok: bool, allow_unconfirmed: bool) -> bool:
        if confirm_ok:
            return True
        if allow_unconfirmed and plausibility_ok:
            return True
        return False
