# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Discovery of custom model implementations shipped as installable packages.

A third-party model is packaged like any Python distribution and advertises itself through
a ``pixano_inference.models`` entry point. Because the package is installed in the same
environment as the server, it is importable both in the ingress process and in every Ray
Serve worker — no source directory needs to be shipped to workers.

Example plugin ``pyproject.toml``::

    [project.entry-points."pixano_inference.models"]
    my_detector = "my_pkg.model"        # importing the module runs @register_model
    # or point directly at the class:
    my_detector = "my_pkg.model:MyDetector"

At startup the server loads all such entry points (importing them registers their models).
The core ships no model implementations and depends on no ML framework: every model --
including the first-party ones (SAM2, CLIP, Grounding DINO, ...) -- is a separate package
discovered this way.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from pixano_inference.models.base import InferenceModel
from pixano_inference.models.registry import ModelClassRegistry


logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pixano_inference.models"

_LOADED = False
# Outcome of the last discovery, kept so that a later "unknown model_class" error can say
# which plugins were found and why one did not load.
_DISCOVERED: dict[str, str] = {}  # entry-point name -> "module[:attr]"
_FAILED: dict[str, str] = {}  # entry-point name -> error


def load_plugin_models() -> dict[str, list[str]]:
    """Discover and import all ``pixano_inference.models`` entry-point plugins.

    Each entry point is loaded (importing its module, which runs the ``@register_model``
    decorators); if it resolves to an :class:`InferenceModel` subclass, it is also
    registered defensively. A failing plugin is logged and skipped rather than aborting
    startup.

    Returns:
        ``{"loaded": [...], "failed": [...]}`` entry-point names.
    """
    loaded: list[str] = []
    failed: list[str] = []
    _DISCOVERED.clear()
    _FAILED.clear()
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = entry_point.load()
            if isinstance(obj, type) and issubclass(obj, InferenceModel):
                ModelClassRegistry.ensure_registered(obj)
            loaded.append(entry_point.name)
            _DISCOVERED[entry_point.name] = entry_point.value
            logger.debug("Loaded model plugin '%s' (%s)", entry_point.name, entry_point.value)
        except Exception as exc:
            logger.warning("Failed to load model plugin '%s' (%s): %s", entry_point.name, entry_point.value, exc)
            failed.append(entry_point.name)
            _FAILED[entry_point.name] = f"{type(exc).__name__}: {exc}"
    return {"loaded": loaded, "failed": failed}


def describe_plugins() -> str:
    """Summarise the last discovery: which model plugins loaded, and which failed and why.

    Returns:
        A one-line description for logs and error messages.
    """
    summary = f"model plugins loaded: {', '.join(sorted(_DISCOVERED)) or 'none'}"
    if _FAILED:
        summary += "; failed to load: " + "; ".join(f"{name} ({error})" for name, error in sorted(_FAILED.items()))
    return summary


def ensure_models_loaded(force: bool = False) -> None:
    """Register all installed models (entry-point plugins).

    Idempotent — safe to call from every entry point that needs the registry populated
    (config resolution, app creation, replica init).

    Args:
        force: Re-run discovery even if it already ran in this process.
    """
    global _LOADED
    if _LOADED and not force:
        return
    load_plugin_models()
    _LOADED = True
    logger.info(
        "%s; registered model classes: %s",
        describe_plugins(),
        ", ".join(sorted(ModelClassRegistry.list_all())) or "none",
    )
