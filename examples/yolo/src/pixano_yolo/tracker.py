# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLO + ByteTrack custom-model plugin: multi-object tracking by detection.

Wraps the track mode of ultralytics (https://docs.ultralytics.com/modes/track) as a
pixano-inference :class:`TrackingModel`. The request carries no prompt: YOLO detects the objects
in each frame and ByteTrack links the detections into tracks, so the model itself decides how many
tracks there are and gives each one its ID.

This module is the target of a ``pixano_inference.models`` entry point declared in this package's
``pyproject.toml``, so installing the package makes ``YOLOByteTrackModel`` available by name.

Install (note: ultralytics is AGPL-3.0)::

    uv sync --project examples/yolo
"""

from __future__ import annotations

import gc
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models.registry import register_model
from pixano_inference.models.tracking import (
    TrackedFrame,
    TrackedObject,
    TrackingInput,
    TrackingInterface,
    TrackingModel,
    TrackingOutput,
)

from .model import class_names_of


logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@register_model("YOLOByteTrackModel")
class YOLOByteTrackModel(TrackingModel):
    """Multi-object tracking by detection: a YOLO detector followed by ByteTrack.

    ``model_params``:

    - ``path`` (str, required): YOLO weights, e.g. ``"yolo26n.pt"``.
    - ``tracker`` (str, optional): ultralytics tracker config. Default ``"bytetrack.yaml"``.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._model: Any = None
        self._tracker: str = "bytetrack.yaml"

    def load_model(self) -> None:
        """Load the YOLO detector from ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLOByteTrackModel. Install with: pip install ultralytics"
            ) from exc

        params = dict(self._config.model_params)
        path = params.pop("path")
        self._tracker = params.pop("tracker", "bytetrack.yaml")

        device = "cpu"
        if self._config.resources.num_gpus > 0:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

        self._model = YOLO(path)
        self._model.to(device)

        logger.info("YOLOByteTrackModel '%s' loaded on %s (tracker=%s)", self.model_name, device, self._tracker)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata."""
        base = super().metadata
        base["path"] = self._config.model_params.get("path")
        base["tracker"] = self._tracker
        return base

    @property
    def interface(self) -> TrackingInterface:
        """Tracking by detection: no prompt, the detector's own classes, a box per track and frame."""
        return TrackingInterface(
            prompts=[],
            prompt_free=True,
            classes="closed",
            class_names=class_names_of(self._model),
            thresholds=["box"],
            outputs=["box", "score", "class"],
        )

    def predict(self, input: TrackingInput) -> TrackingOutput:
        """Detect the objects of each frame and link them into tracks.

        Args:
            input: Tracking input with the video only, and optionally ``classes`` (class names to
                keep) and ``box_threshold`` (detector confidence floor).

        Returns:
            Tracking output: for each frame, the box, score and class of each track alive in it.

        Raises:
            ValueError: If the request names objects or carries prompts (this model creates its own
                tracks), or asks for a class the detector does not know.
        """
        has_prompts = any(prompt is not None for prompt in (input.keyframes, input.points, input.labels, input.boxes))
        if input.objects_ids or has_prompts:
            raise ValueError("YOLOByteTrackModel tracks by detection: send the video without object IDs or prompts.")

        options: dict[str, Any] = {"tracker": self._tracker, "verbose": False}
        class_ids = self._class_ids(input.classes)
        if class_ids is not None:
            options["classes"] = class_ids
        # Track mode keeps the detector confidence low (0.1) on purpose: ByteTrack associates the
        # low-score boxes too. A request may raise that floor.
        if input.box_threshold is not None:
            options["conf"] = input.box_threshold

        return TrackingOutput(
            frames=[
                self._tracked_frame(index, result) for index, result in enumerate(self._track(input.video, options))
            ]
        )

    def _class_ids(self, classes: list[str] | str | None) -> list[int] | None:
        """Translate class names into the detector's class indices."""
        if classes is None:
            return None
        requested = [classes] if isinstance(classes, str) else list(classes)
        index_of = {name: index for index, name in self._model.names.items()}
        unknown = [name for name in requested if name not in index_of]
        if unknown:
            raise ValueError(f"Unknown class name(s) {unknown}. This detector knows: {sorted(index_of)}.")
        return [index_of[name] for name in requested]

    def _track(self, video: Any, options: dict[str, Any]) -> Iterator[Any]:
        """Yield the ultralytics result of each frame, in order, with one tracker for the request.

        ``persist=False`` on the first frame starts fresh trackers (track IDs restart at 1);
        ``persist=True`` on the following frames continues the sequence.
        """
        from pixano_inference.utils.media import convert_string_to_image, convert_string_video_to_bytes_or_path

        if isinstance(video, list):
            frames = video
        else:
            source = convert_string_video_to_bytes_or_path(video)
            if isinstance(source, Path) and source.is_dir():
                frames = sorted(path for path in source.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
            else:
                yield from self._track_video_file(source, options)
                return

        for index, frame in enumerate(frames):
            image = convert_string_to_image(frame)
            yield self._model.track(image, persist=index > 0, **options)[0]

    def _track_video_file(self, source: Any, options: dict[str, Any]) -> Iterator[Any]:
        """Track a single video file, given as a path or as its bytes."""
        with tempfile.TemporaryDirectory() as directory:
            if isinstance(source, bytes):
                path = Path(directory) / "video.mp4"
                path.write_bytes(source)
            else:
                path = source
            yield from self._model.track(source=str(path), stream=True, persist=False, **options)

    @staticmethod
    def _tracked_frame(frame_index: int, result: Any) -> TrackedFrame:
        """Turn the ultralytics result of one frame into a tracked frame."""
        boxes = result.boxes
        if boxes is None or boxes.id is None:  # nothing is tracked in this frame
            return TrackedFrame(frame_index=frame_index)

        return TrackedFrame(
            frame_index=frame_index,
            objects=[
                TrackedObject(
                    track_id=int(track_id),
                    box=[round(float(coordinate), 1) for coordinate in xyxy],
                    score=float(score),
                    class_name=result.names[int(class_id)],
                )
                for track_id, xyxy, score, class_id in zip(
                    boxes.id.cpu().numpy(),
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                )
            ],
        )

    def unload(self) -> None:
        """Free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
