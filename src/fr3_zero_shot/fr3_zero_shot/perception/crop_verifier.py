from __future__ import annotations


class CropVerifier:
    def verify_label(self, *, expected: str, observed_text: str) -> bool:
        expected_tokens = set(str(expected).lower().split())
        observed_tokens = set(str(observed_text).lower().split())
        return bool(expected_tokens and expected_tokens.issubset(observed_tokens))

