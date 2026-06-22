"""Deterministic parasite detector for red blood cell crops."""

from posthoc.parasite_detector.config import DEFAULT_CONFIG
from posthoc.parasite_detector.detector import detect_parasites

__all__ = ["DEFAULT_CONFIG", "detect_parasites"]
