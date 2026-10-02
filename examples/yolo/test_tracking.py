# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

r"""Test script for YOLO + ByteTrack tracking with pixano-inference.

Demonstrates how to:
1. Connect to the pixano-inference server via the client
2. Track every object of a clip without any prompt (tracking by detection)
3. Read the result frame by frame and track by track

Prerequisites:
- Start the server with the tracking config:
    uv run --project examples/yolo pixano-inference --config examples/yolo/config_tracking.py
Usage:
    python examples/yolo/test_tracking.py \
        [--server-url URL] [--frames DIR] [--model-name NAME] \
        [--max-frames 30] [--classes person] [--threshold 0.25] [--output tracking.png]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

from pixano_inference.client import PixanoInferenceClient, PixanoInferenceError
from pixano_inference.schemas import TrackingOutput, TrackingRequestV1


DEFAULT_SERVER_URL = "http://localhost:7463"
DEFAULT_MODEL_NAME = "yolo-bytetrack"
DEFAULT_FRAMES = Path(__file__).resolve().parents[2] / "docs" / "assets" / "examples" / "sam2" / "bedroom"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def frame_to_base64(frame_path: Path) -> str:
    """Convert a frame file to a base64 data-URI string.

    Args:
        frame_path: Path to the image file.

    Returns:
        Base64 encoded data-URI string.
    """
    mime = "png" if frame_path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(frame_path.read_bytes()).decode('utf-8')}"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def save_contact_sheet(frames: list[Path], output: TrackingOutput, destination: Path, columns: int = 4) -> None:
    """Draw the tracked boxes on a few frames and save them side by side.

    Args:
        frames: The frame files, in the order they were sent.
        output: The tracking result.
        destination: Where to save the image.
        columns: Number of frames to draw, spread over the clip.
    """
    from PIL import Image, ImageDraw

    palette = [(230, 60, 60), (40, 130, 240), (50, 180, 90), (240, 170, 30), (160, 80, 220), (20, 190, 190)]
    by_frame = {frame.frame_index: frame for frame in output.frames}
    step = max(1, (len(frames) - 1) // max(1, columns - 1))
    picked = list(range(0, len(frames), step))[:columns]

    tiles = []
    for frame_index in picked:
        tile = Image.open(frames[frame_index]).convert("RGB")
        draw = ImageDraw.Draw(tile)
        for tracked in by_frame[frame_index].objects if frame_index in by_frame else []:
            if tracked.box is None:  # a mask-only tracker returns no box
                continue
            x1, y1, x2, y2 = tracked.box
            color = palette[tracked.track_id % len(palette)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            label = f"#{tracked.track_id} {tracked.class_name} {tracked.score:.2f}"
            top = max(0, y1 - 14)
            draw.rectangle((x1, top, x1 + 7 * len(label), top + 14), fill=color)
            draw.text((x1 + 2, top + 1), label, fill=(255, 255, 255))
        draw.text((8, 8), f"frame {frame_index}", fill=(255, 255, 0))
        tiles.append(tile)

    sheet = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    sheet.save(destination)


async def main() -> None:
    """Run the YOLO + ByteTrack tracking test."""
    parser = argparse.ArgumentParser(description="Test YOLO + ByteTrack multi-object tracking")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES, help="Directory of video frames")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--max-frames", type=int, default=30, help="Number of frames to send (default: 30)")
    parser.add_argument("--classes", nargs="*", default=None, help="Class names to track (default: all)")
    parser.add_argument("--threshold", type=float, default=None, help="Detector confidence floor (default: model's)")
    parser.add_argument("--output", type=Path, default=Path("tracking_result.png"), help="Annotated frames to save")
    args = parser.parse_args()

    # --- Connect ---
    print_section("YOLO + ByteTrack Tracking Test")
    print(f"\nServer URL: {args.server_url}")

    client = PixanoInferenceClient.connect(args.server_url)
    try:
        models = await client.list_models()
    except PixanoInferenceError:
        print("\nERROR: Server is not running!")
        print("Start the server with:")
        print("  uv run --project examples/yolo pixano-inference --config examples/yolo/config_tracking.py")
        sys.exit(1)

    if not any(m.name == args.model_name for m in models):
        available = ", ".join(f"{m.name} ({m.capability})" for m in models) or "none"
        print(f"\nERROR: Model '{args.model_name}' not found. Available: {available}")
        sys.exit(1)

    # --- Prepare frames ---
    print_section("Preparing Frames")
    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.max_frames]
    if not frames:
        print(f"ERROR: No frame found in {args.frames}")
        sys.exit(1)
    print(f"Using {len(frames)} frames from {args.frames}")

    # --- Track: the request names no object, the model creates the tracks ---
    print_section("Tracking")
    request = TrackingRequestV1(
        model=args.model_name,
        video=[frame_to_base64(frame) for frame in frames],
        classes=args.classes,
        box_threshold=args.threshold,
    )
    try:
        response = await client.tracking(request)
    except PixanoInferenceError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    finally:
        await client.aclose()

    output = response.data
    print(f"Status: {response.status}")
    print(f"Processing time: {response.processing_time:.3f}s")
    print(f"Frames: {len(output.frames)}")

    # By frame: what the model returns.
    first = output.frames[0]
    print(f"\nFrame {first.frame_index}:")
    for tracked in first.objects:
        print(f"  track #{tracked.track_id} class={tracked.class_name} score={tracked.score:.3f} box={tracked.box}")

    # By track: the same result as trajectories.
    tracks = output.tracks()
    print(f"\nTracks: {len(tracks)}")
    for track_id, track in sorted(tracks.items()):
        frame_indexes = [frame_index for frame_index, _ in track]
        class_name = track[0][1].class_name
        print(
            f"  #{track_id} class={class_name} frames {frame_indexes[0]}-{frame_indexes[-1]} "
            f"({len(track)} of {len(frames)})"
        )

    save_contact_sheet(frames, output, args.output)
    print(f"\nAnnotated frames saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
