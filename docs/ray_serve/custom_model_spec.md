<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Custom Model Package Specification

This is the contract between a model package and the Pixano Inference core. The core ships no
model: every model, first-party or yours, is a package that follows this specification and is
plugged in at runtime. MUST, SHOULD and MAY are used in the RFC 2119 sense.

The reference implementation is
[`examples/numpy_detector`](https://github.com/pixano/pixano-inference/tree/main/examples/numpy_detector),
a complete, framework-free detector; each rule below says how it meets it.
[`examples/yolo`](https://github.com/pixano/pixano-inference/tree/main/examples/yolo) shows the
same contract around a third-party framework (ultralytics). The
[Custom Models Guide](custom_models.md) is the tutorial version of this page.

```
examples/numpy_detector/
├── pyproject.toml                       # metadata, dependencies, entry point   (§1)
├── uv.lock                              # the package's own lock file            (§1.3)
├── LICENSE
├── src/pixano_numpy_detector/
│   ├── __init__.py
│   └── model.py                         # params + model class                   (§2, §3)
└── tests/test_model.py                  # discovery, params, predict             (§6)
```

## 1. Package

**1.1 One distribution per model.** A model package MUST be an installable Python
distribution with its own `pyproject.toml`. It MAY contain several models of one family (the
SAM2 package registers an image and a video model). Its `requires-python` MUST overlap the
core's (`>=3.10,<3.14`).

**1.2 Dependencies.** The package MUST declare `pixano-inference` and every library it
imports, its ML framework included. It MUST NOT rely on a dependency another package happens
to install, and MUST NOT import another model package. PyTorch models MAY depend on
`pixano-inference-torch` for device, dtype and tensor helpers.

**1.3 Environment.** The package SHOULD keep its own `uv.lock`, so it is developed, tested
and served from its own environment, independent of the core's repository and of other
models. `[tool.uv.sources]` MAY point `pixano-inference` at a clone or a git ref of the core
for development.

**1.4 Entry point.** The package MUST declare one entry point per model module (or class) in
the `pixano_inference.models` group. The target is `module` or `module:Class`; importing it
MUST register the model class and its params (§2.2, §3). The server imports every entry point
of the group at startup, and Ray Serve workers import the model's module from the same
installed package, so nothing else is needed for a model to be found by name.

In the example (`pyproject.toml`):

```toml
[project]
name = "pixano-numpy-detector"
requires-python = ">=3.10,<3.14"
dependencies = ["pixano-inference >= 0.6.0, < 0.7.0", "numpy >= 1.26.0, < 3.0.0", "Pillow >= 9.0.0", "pydantic >= 2.0.0, < 3.0.0"]

[project.entry-points."pixano_inference.models"]
numpy_detector = "pixano_numpy_detector.model"
```

## 2. Model class

**2.1 Base class.** A model MUST subclass exactly one capability base from
`pixano_inference.models`. The base fixes the capability, the HTTP route and the typed
input/output:

| Base class          | Capability     | Route                        | Input → Output                             |
| ------------------- | -------------- | ---------------------------- | ------------------------------------------ |
| `DetectionModel`    | `detection`    | `/v1/inference/detection`    | `DetectionInput` → `DetectionOutput`       |
| `SegmentationModel` | `segmentation` | `/v1/inference/segmentation` | `SegmentationInput` → `SegmentationOutput` |
| `TrackingModel`     | `tracking`     | `/v1/inference/tracking`     | `TrackingInput` → `TrackingOutput`         |
| `VLMModel`          | `vlm`          | `/v1/inference/vlm`          | `VLMInput` → `VLMOutput`                   |
| `NERModel`          | `ner`          | `/v1/inference/ner`          | `NERInput` → `NEROutput`                   |
| `EmbeddingModel`    | `embedding`    | `/v1/inference/embedding`    | `EmbeddingInput` → `EmbeddingOutput`       |

A subclass of `InferenceModel` that is not one of these is rejected by `ModelConfig`.

**2.2 Registration.** The class MUST be decorated with `@register_model("Name")`. `Name` is
what configs put in `model_class`; it MUST be unique across the installed packages (a second
registration of the same name raises, and that package fails to load).

**2.3 Construction.** The constructor is `__init__(self, config: ModelDeploymentConfig)`. An
override MUST call `super().__init__(config)` and MUST NOT load weights or import the
framework: the replica constructs the model and immediately calls `load_model()` (§2.4), so
the constructor only initializes attributes.

**2.4 `load_model()`.** Called once per replica, in the Serve actor's `__init__`. It MUST leave
the model ready for `predict()`. It reads its settings from `self.config.model_params` (a
`dict`, see §3) and its resources from `self.config.resources` (`num_gpus`, `num_cpus`). A
failure here fails the deployment: at startup with `--config`, or as the error of
`POST /v1/models`.

**2.5 `predict(input)`.** MUST accept the base class's input type and return its output type,
synchronously. Calls on one replica are serialized by the server, so the model need not be
thread-safe. Anything raised fails that request with a `500` error envelope (the exception is
logged server-side, its text is not returned); a call that outlives the deployment's
`timeout_s` fails with `504`.

**2.6 `predict_batch(inputs)`.** MAY be overridden. It is used instead of `predict()` only when
the deployment sets `max_batch_size > 1`, and MUST return one output per input, in order. The
default runs `predict()` sequentially.

**2.7 `unload()` and `metadata`.** `unload()` MAY release resources; the server calls it when a
replica is torn down. The `metadata` property MAY be extended with a `dict`; it is served by
the replica's `get_metadata()` handle method and is not part of the HTTP API.

**2.8 Lazy framework imports.** Importing the package MUST NOT import the ML framework.
Import it inside `load_model()` / `predict()` / `unload()`. Discovery imports every installed
model package, so an eager import slows every start; and a broken framework install would then
hide the whole package instead of failing one deployment with a clear error.

**2.9 Media.** Image fields (`image` in the detection, segmentation and embedding inputs) are a
URL, a `data:` URI or a server-local path. The model MUST resolve them with
`pixano_inference.utils.media.convert_string_to_image` (videos:
`convert_string_video_to_bytes_or_path`), which enforces the server's media policy: private
addresses are refused for URLs, local paths are allowed only under
`PIXANO_INFERENCE_MEDIA_ROOTS`, and sizes are capped.

**2.10 Device.** `self.config.resources.num_gpus > 0` means the deployment was given a GPU;
the model chooses the device from it. `pixano_inference_torch.resolve_device(self.config)`
does this for PyTorch (CUDA, then Apple MPS, else CPU).

In the example (`model.py`), 2.1 to 2.5 and 2.9; no framework, so 2.8 and 2.10 are moot:

```python
@register_model("NumpyDetector")
class NumpyDetector(DetectionModel):
    def load_model(self) -> None:
        self._threshold = int(self.config.model_params.get("threshold", 20))

    def predict(self, input: DetectionInput) -> DetectionOutput:
        image = np.asarray(convert_string_to_image(input.image), dtype=np.int16)
        ...
        return DetectionOutput(boxes=[box], scores=[score], classes=["object"], masks=None)
```

In the yolo example, 2.8 and 2.10: `from ultralytics import YOLO` sits inside `load_model()`,
which picks `"cuda"` only when `num_gpus > 0` and CUDA is available, and `unload()` frees the
model and the CUDA cache.

## 3. Parameters

**3.1 Typed params.** A model SHOULD define a params class: a subclass of
`pixano_inference.configs.BaseModelParams`, registered with
`@register_model_params("Name")` under the model's registered name. `BaseModelParams` forbids
unknown fields and requires `path` (a checkpoint id or location; give it a default if the
model has none). With a params class registered, `ModelConfig(model_params={...})` validates
the dict, applies the defaults and rejects typos, and a config that omits `model_params`
entirely gets the defaults.

**3.2 Without typed params.** `model_params` is passed through as given. `load_model()` then
owns validation.

**3.3 In the replica.** `self.config.model_params` is always a plain `dict`: the params dumped
by the config, never the params object.

In the example:

```python
@register_model_params("NumpyDetector")
class NumpyDetectorParams(BaseModelParams):
    path: str = "numpy-detector"           # no checkpoint: names the deployment when `name` is omitted
    threshold: int = Field(default=20, ge=0)
```

## 4. Configuration

A deployment config is a Python file with a `models` list of `ModelConfig`; the server loads
it with `--config`, or receives the same fields through `POST /v1/models` after startup.

| Field          | Meaning                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `name`         | Unique deployment name (the `model` a client sends). Defaults to the last segment of `path`.                 |
| `model_class`  | The registered name (§2.2), as a string: no import needed.                                                   |
| `model_params` | A params object or a dict, validated as in §3.                                                               |
| `deployment`   | `DeploymentConfig`: `num_gpus`, `num_cpus`, `min_replicas`, `max_replicas`, `max_batch_size`, `timeout_s`, … |

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="numpy-detector",
        model_class="NumpyDetector",
        model_params={"threshold": 20},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=1),
    ),
]
```

## 5. Distribution

**5.1 Installed, not shipped.** The package MUST be installed in the environment the server
runs in (Ray Serve workers import it from there). Loose source directories and `PYTHONPATH`
are not an integration mechanism.

**5.2 Any source.** Publishing is optional. The package installs from PyPI or a private index,
from a git URL (`#subdirectory=` for a monorepo), from a local directory, or from a directory
of wheels; `uv pip install` also follows the package's `[tool.uv.sources]`.

**5.3 Core version.** The package SHOULD constrain `pixano-inference` to the versions whose
contract it was written against (`>= 0.6.0, < 0.7.0` today), as the first-party packages do.

## 6. Conformance

A package conforms when the following hold. The example's `tests/test_model.py` covers the
first three; the yolo example's `tests/test_discovery.py` covers the fourth.

| Check                                       | How                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Discovered through its entry point          | `load_plugin_models()["loaded"]` contains the entry point; `ModelClassRegistry.has("Name")` |
| Params validate and default                 | `ModelConfig(name="x", model_class="Name").to_deployment_config().model_params`             |
| `predict()` returns the capability's output | Call the model directly on a small input                                                    |
| Importing the package loads no framework    | `python -c "import sys, my_pkg; print([m for m in ('torch',) if m in sys.modules])"`        |
| Lock file matches the metadata              | `uv lock --check --project <package>`                                                       |
| Serves end to end                           | Start the server from the package's environment and call the route                          |

```bash
uv run --project examples/numpy_detector pytest examples/numpy_detector/tests
uv lock --check --project examples/numpy_detector
uv run --project examples/numpy_detector pixano-inference --config models.py
```
