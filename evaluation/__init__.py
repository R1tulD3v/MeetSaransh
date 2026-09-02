"""Retrieval-quality evaluation for the "Ask your meetings" pipeline.

Kept out of `app/` on purpose: this is a measurement tool, not part of the service, and
nothing the server imports at runtime should depend on it.

(Named `evaluation` rather than `eval` so it never shadows the builtin.)
"""
