from __future__ import annotations


def should_emit_debug_overlay(debug_save_detection: bool, debug_publish_image: bool) -> bool:
    return bool(debug_save_detection or debug_publish_image)
