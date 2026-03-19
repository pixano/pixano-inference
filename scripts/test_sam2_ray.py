# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Test script for SAM2 inference with Ray backend using the PixanoInferenceClient.

This script demonstrates how to:
1. Connect to the pixano-inference Ray backend server via the client
2. List deployed models
3. Perform inference with point and box prompts using typed requests/responses
4. Decode and inspect the resulting masks

Prerequisites:
- Start the Ray backend server with models configured in your YAML config:

    pixano-inference --host 0.0.0.0 --port 8000 --backend ray

Usage:
    python scripts/test_sam2_ray.py [--server-url URL] [--image PATH] [--model-name NAME]
"""

import argparse
import asyncio
import base64
import sys
from pathlib import Path

import numpy as np
import requests
from fastapi import HTTPException

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest


# Default configuration
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "sam2-image"


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string.
    """
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64_data}"


def create_sample_image(width: int = 640, height: int = 480) -> str:
    """Create a sample test image as base64.

    Creates a simple image with colored shapes for testing.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        Base64 encoded PNG image.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed. Please install with: pip install Pillow")
        sys.exit(1)

    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw a red rectangle (object 1)
    draw.rectangle([100, 100, 300, 250], fill=(200, 50, 50))

    # Draw a blue circle (object 2)
    draw.ellipse([350, 150, 550, 350], fill=(50, 50, 200))

    # Draw a green triangle (object 3)
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


async def main():
    """Main function to run SAM2 inference tests."""
    parser = argparse.ArgumentParser(description="Test SAM2 inference with Ray backend")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (if not provided, creates a test image)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to use for inference (default: {DEFAULT_MODEL_NAME})",
    )

    args = parser.parse_args()

    # 1. Connect to server
    print_section("SAM2 Ray Backend Test Script")
    print(f"\nServer URL: {args.server_url}")

    try:
        client = PixanoInferenceClient.connect(args.server_url)
    except requests.ConnectionError:
        print("\nERROR: Server is not running!")
        print("Start the server with:")
        print("  pixano-inference --host 0.0.0.0 --port 8000 --backend ray")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to connect: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"Connected! GPUs: {client.num_gpus}")

    # 2. List current models
    print_section("Deployed Models")
    models = await client.list_models()
    if models:
        for model in models:
            print(f"  - {model.name} (task={model.task})")
    else:
        print("  No models deployed. Configure models in your Ray YAML config.")
        sys.exit(1)

    # Check requested model exists
    if not any(m.name == args.model_name for m in models):
        available = ", ".join(m.name for m in models)
        print(f"\nERROR: Model '{args.model_name}' not found. Available: {available}")
        print("Use --model-name to pick one.")
        sys.exit(1)

    # 3. Prepare image
    print_section("Preparing Image")
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: Image file not found: {args.image}")
            sys.exit(1)
        print(f"Using image: {args.image}")
        image_b64 = image_to_base64(image_path)
    else:
        print("Creating test image with shapes...")
        image_b64 = create_sample_image()

    # 4. Run inference tests
    print_section("Running Inference Tests")
    model_name = args.model_name

    # Test 1: Point prompt (center of red rectangle)
    print("\n--- Test 1: Single point prompt ---")
    try:
        request = SegmentationRequest(
            model=model_name,
            image=image_b64,
            points=[[[200, 175]]],
            labels=[[1]],
        )
        result = await client.segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Number of prompts: {len(result.data.masks)}")
        print(f"Masks per prompt: {len(result.data.masks[0])}")
        print(f"Scores: {result.data.scores.to_numpy()}")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    # Test 2: Box prompt (around blue circle)
    print("\n--- Test 2: Box prompt ---")
    try:
        request = SegmentationRequest(
            model=model_name,
            image=image_b64,
            boxes=[[350, 150, 550, 350]],
        )
        result = await client.segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Number of prompts: {len(result.data.masks)}")
        print(f"Masks per prompt: {len(result.data.masks[0])}")
        print(f"Scores: {result.data.scores.to_numpy()}")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    # Test 3: Multiple points (foreground + background)
    print("\n--- Test 3: Foreground + background points ---")
    try:
        request = SegmentationRequest(
            model=model_name,
            image=image_b64,
            points=[[[200, 175], [450, 250]]],
            labels=[[1, 0]],
            multimask_output=False,
        )
        result = await client.segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Number of prompts: {len(result.data.masks)}")
        print(f"Masks per prompt: {len(result.data.masks[0])}")
        print(f"Scores: {result.data.scores.to_numpy()}")

        # Decode mask using CompressedRLE.to_mask()
        mask = result.data.masks[0][0].to_mask()
        print(f"Mask shape: {mask.shape}")
        print(f"Mask coverage: {np.sum(mask) / mask.size * 100:.2f}%")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    # Test 4: Multiple prompts (batch)
    print("\n--- Test 4: Multiple prompts (batch) ---")
    try:
        request = SegmentationRequest(
            model=model_name,
            image=image_b64,
            points=[
                [[200, 175]],  # Prompt 1: red rectangle
                [[450, 250]],  # Prompt 2: blue circle
                [[200, 400]],  # Prompt 3: green triangle
            ],
            labels=[
                [1],
                [1],
                [1],
            ],
        )
        result = await client.segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Number of prompts: {len(result.data.masks)}")
        for i, masks in enumerate(result.data.masks):
            print(f"  Prompt {i + 1}: {len(masks)} masks")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    # Test 5: Get embeddings for reuse
    print("\n--- Test 5: Get embeddings for reuse ---")
    try:
        request = SegmentationRequest(
            model=model_name,
            image=image_b64,
            points=[[[200, 175]]],
            labels=[[1]],
            return_image_embedding=True,
        )
        result = await client.segmentation(request)
        print(f"Status: {result.status}")
        print(f"Processing time: {result.processing_time:.3f}s")
        has_embedding = result.data.image_embedding is not None
        has_features = result.data.high_resolution_features is not None
        print(f"Image embedding returned: {has_embedding}")
        print(f"High-res features returned: {has_features}")
        if has_embedding:
            print(f"Embedding shape: {result.data.image_embedding.shape}")
    except HTTPException as e:
        print(f"ERROR: {e.detail}")

    print_section("Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
