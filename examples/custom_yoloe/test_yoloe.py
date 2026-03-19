# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

r"""Test script for YOLOE instance segmentation with pixano-inference.

Demonstrates how to:
1. Connect to the pixano-inference server via the client
2. List deployed models
3. Run instance segmentation with open-vocab and closed-vocab modes
4. Decode and inspect the resulting boxes and masks

Prerequisites:
- Start the server with the YOLOE config:

    pixano-inference \
        --module-path examples \
        --config examples/custom_yoloe/config.py

Usage:
    python examples/custom_yoloe/test_yoloe.py \
        [--server-url URL] [--image PATH] [--model-name NAME] \
        [--classes "cat,dog"] [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

import numpy as np
import requests
from fastapi import HTTPException

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import DetectionRequest


DEFAULT_SERVER_URL = "http://localhost:7463"
DEFAULT_MODEL_NAME = "yoloe"


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to a base64 data-URI string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded data-URI string.
    """
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64_data}"


def create_sample_image(width: int = 640, height: int = 480) -> str:
    """Create a sample test image as a base64 data-URI.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        Base64 encoded PNG image.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 250], fill=(200, 50, 50))
    draw.ellipse([350, 150, 550, 350], fill=(50, 50, 200))
    draw.polygon([(200, 350), (100, 450), (300, 450)], fill=(50, 200, 50))

    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


async def main() -> None:
    """Run YOLOE instance segmentation tests."""
    parser = argparse.ArgumentParser(description="Test YOLOE instance segmentation")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (creates test image if omitted)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names for open-vocab mode")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection confidence threshold (default: 0.3)")
    args = parser.parse_args()

    # --- Connect ---
    print_section("YOLOE Instance Segmentation Test")
    print(f"\nServer URL: {args.server_url}")

    try:
        client = PixanoInferenceClient.connect(args.server_url)
    except requests.ConnectionError:
        print("\nERROR: Server is not running!")
        print("Start the server with:")
        print(
            "  pixano-inference "
            "--module-path examples --config examples/custom_yoloe/config.py"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to connect: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"Connected! GPUs: {client.num_gpus}")

    # --- List models ---
    print_section("Deployed Models")
    models = await client.list_models()
    if models:
        for model in models:
            print(f"  - {model.name} (task={model.task})")
    else:
        print("  No models deployed.")
        sys.exit(1)

    if not any(m.name == args.model_name for m in models):
        available = ", ".join(m.name for m in models)
        print(f"\nERROR: Model '{args.model_name}' not found. Available: {available}")
        sys.exit(1)

    # --- Prepare image ---
    print_section("Preparing Image")
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: Image file not found: {args.image}")
            sys.exit(1)
        print(f"Using image: {args.image}")
        image_b64 = image_to_base64(image_path)
    else:
        print("Creating synthetic test image...")
        image_b64 = create_sample_image()

    model_name = args.model_name

    # --- Test 1: Closed-vocab (prompt-free) ---
    print_section("Test 1: Closed-Vocab Detection (no classes)")
    try:
        request = DetectionRequest(
            model=model_name,
            image=image_b64,
            classes=None,
            box_threshold=args.threshold,
        )
        result = await client.instance_segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Detections: {len(result.data.boxes)}")
        for i, (box, score, cls) in enumerate(zip(result.data.boxes, result.data.scores, result.data.classes)):
            print(f"  [{i}] class={cls}, score={score:.3f}, box={box}")
        if result.data.masks:
            print(f"Masks returned: {len(result.data.masks)}")
            for i, mask_rle in enumerate(result.data.masks):
                mask = mask_rle.to_mask()
                coverage = np.sum(mask) / mask.size * 100
                print(f"  [{i}] mask shape={mask.shape}, coverage={coverage:.2f}%")
        else:
            print("No masks returned (detection-only mode).")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    # --- Test 2: Open-vocab (with classes) ---
    classes = args.classes.split(",") if args.classes else ["rectangle", "circle", "triangle"]
    print_section(f"Test 2: Open-Vocab Detection (classes={classes})")
    try:
        request = DetectionRequest(
            model=model_name,
            image=image_b64,
            classes=classes,
            box_threshold=args.threshold,
        )
        result = await client.instance_segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Detections: {len(result.data.boxes)}")
        for i, (box, score, cls) in enumerate(zip(result.data.boxes, result.data.scores, result.data.classes)):
            print(f"  [{i}] class={cls}, score={score:.3f}, box={box}")
        if result.data.masks:
            print(f"Masks returned: {len(result.data.masks)}")
            for i, mask_rle in enumerate(result.data.masks):
                mask = mask_rle.to_mask()
                coverage = np.sum(mask) / mask.size * 100
                print(f"  [{i}] mask shape={mask.shape}, coverage={coverage:.2f}%")
        else:
            print("No masks returned (detection-only mode).")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    print_section("Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
