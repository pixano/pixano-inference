# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The deprecated ``pixano_inference_client`` alias re-exports the core, name for name.

Every name the 0.1.0 distribution exposed must still resolve, and to the very object the core
defines: two copies of a wire type would not compare equal or validate each other.
"""

import importlib
import subprocess
import sys

import pytest


# Public names of each 0.1.0 submodule, and the core module that now defines them.
SUBMODULES = {
    "base": ("pixano_inference.schemas.base", ["BaseRequest", "BaseResponse", "CamelModel"]),
    "nd_array": ("pixano_inference.schemas.nd_array", ["NDArray", "NDArrayFloat"]),
    "rle": ("pixano_inference.schemas.rle", ["CompressedRLE", "mask_to_rle", "rle_to_mask"]),
    "models_info": ("pixano_inference.schemas.models", ["ModelInfo"]),
    "detection": ("pixano_inference.schemas.detection", ["DetectionInput", "DetectionOutput"]),
    "embedding": ("pixano_inference.schemas.embedding", ["EmbeddingInput", "EmbeddingOutput"]),
    "ner": ("pixano_inference.schemas.ner", ["NEREntity", "NERInput", "NEROutput"]),
    "segmentation": ("pixano_inference.schemas.segmentation", ["SegmentationInput", "SegmentationOutput"]),
    "vlm": ("pixano_inference.schemas.vlm", ["UsageInfo", "VLMInput", "VLMOutput"]),
    "tracking": (
        "pixano_inference.schemas.tracking",
        [
            "TrackingBoxPrompt",
            "TrackingInput",
            "TrackingInterval",
            "TrackingKeyframe",
            "TrackingOutput",
            "TrackingPointPrompt",
        ],
    ),
    "inference": (
        "pixano_inference.schemas.inference",
        [
            "DetectionRequest",
            "DetectionResponse",
            "EmbeddingRequest",
            "EmbeddingResponse",
            "NERRequest",
            "NERResponse",
            "SegmentationRequest",
            "SegmentationResponse",
            "TrackingRequest",
            "TrackingResponse",
            "VLMRequest",
            "VLMResponse",
        ],
    ),
    "v1": (
        "pixano_inference.schemas.v1",
        [
            "DeployModelRequest",
            "JobStatus",
            "ModelStatusInfo",
            "TrackingKeyframeV1",
            "TrackingPrompts",
            "TrackingRequestV1",
        ],
    ),
    "client": (
        "pixano_inference.client",
        [
            "DEFAULT_TIMEOUT",
            "DEPLOY_TIMEOUT",
            "TRACKING_TIMEOUT",
            "PixanoInferenceClient",
            "PixanoInferenceError",
            "SyncPixanoInferenceClient",
        ],
    ),
}

# ``pixano_inference_client.__all__`` as published in 0.1.0.
TOP_LEVEL_NAMES = sorted(
    name
    for submodule, (_, names) in SUBMODULES.items()
    for name in names
    if not (submodule == "client" and name.endswith("_TIMEOUT"))
)


def test_importing_the_alias_warns_about_the_deprecation():
    script = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import pixano_inference_client\n"
        "messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]\n"
        "assert any('pixano_inference.client' in m for m in messages), messages\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_top_level_names_are_the_0_1_0_names_and_the_core_objects():
    import pixano_inference_client

    import pixano_inference.client
    import pixano_inference.schemas

    assert sorted(pixano_inference_client.__all__) == TOP_LEVEL_NAMES
    for name in pixano_inference_client.__all__:
        core_module = (
            pixano_inference.client if "Client" in name or name.endswith("Error") else pixano_inference.schemas
        )
        assert getattr(pixano_inference_client, name) is getattr(core_module, name), name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("submodule", sorted(SUBMODULES))
def test_submodule_exposes_the_core_objects(submodule):
    core_path, names = SUBMODULES[submodule]
    alias_module = importlib.import_module(f"pixano_inference_client.{submodule}")
    core_module = importlib.import_module(core_path)

    assert sorted(alias_module.__all__) == sorted(names)
    for name in names:
        assert getattr(alias_module, name) is getattr(core_module, name), name


def test_importing_the_alias_loads_no_server_module():
    heavy = [
        "ray",
        "fastapi",
        "starlette",
        "uvicorn",
        "typer",
        "torch",
        "pixano_inference.ray",
        "pixano_inference.api",
    ]
    script = (
        "import sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import pixano_inference_client\n"
        f"leaked = [m for m in {heavy!r} if m in sys.modules]\n"
        "assert not leaked, 'the alias pulled in server modules: ' + repr(leaked)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
