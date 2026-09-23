<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

<div align="center">

<img src="https://raw.githubusercontent.com/pixano/pixano/main/docs/assets/pixano_wide.png" alt="Pixano" height="100"/>

<br/>
<br/>

**Pixano-Inference is an open-source inference library for Pixano.**

**_Under active development, subject to API change_**

[![GitHub version](https://img.shields.io/github/v/release/pixano/pixano-inference?label=release&logo=github)](https://github.com/pixano/pixano-inference/releases)
[![PyPI version](https://img.shields.io/pypi/v/pixano-inference?color=blue&label=release&logo=pypi&logoColor=white)](https://pypi.org/project/pixano-inference/)
[![Tests](https://img.shields.io/github/actions/workflow/status/pixano/pixano-inference/test_back.yml?branch=develop)](https://github.com/pixano/pixano-inference/actions/workflows/test_back.yml)
[![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fpixano.github.io%2F&up_message=online&down_message=offline&label=docs)](https://pixano.github.io)
[![Python version](https://img.shields.io/pypi/pyversions/pixano-inference?color=important&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

</div>

<hr />

# Pixano-Inference

## Context

This library provides a Ray Serve-based inference server for multimodal AI
tasks. It was first built to support the
[Pixano](https://pixano.github.io/pixano/latest/) AI-powered annotation tool
and exposes typed deployment configs, a Python client, and a REST API for
running deployed models.

## Installation

To install the library, simply execute the following command

```bash
pip install pixano-inference
```

The core is framework-free: it ships no model and depends on no ML framework. Models are
separate packages, each bringing its own framework, and the server discovers every installed
one automatically — install those you need alongside the core:

```bash
pip install pixano-inference-sam pixano-inference-grounding-dino
```

| Package                             | Models                                             |
| ----------------------------------- | -------------------------------------------------- |
| `pixano-inference-sam`              | SAM2 image segmentation and video tracking         |
| `pixano-inference-clip`             | CLIP-style image/text embeddings (MobileCLIP2)     |
| `pixano-inference-grounding-dino`   | Grounding DINO zero-shot detection                 |
| `pixano-inference-transformers-vlm` | Vision-language models through Hugging Face        |
| `pixano-inference-vllm`             | Vision-language models served by vLLM (Linux, GPU) |

Each package under `packages/` is self-contained, with its own `pyproject.toml` and `uv.lock`.
Its environment holds the core plus that model, so from a clone the server runs from it:

```bash
cd packages/pixano-inference-sam
uv sync
uv run pixano-inference --config models.py
```

If you want to dynamically make changes to the library to develop and test, make a dev install by cloning the repo and executing the following commands

```bash
cd pixano-inference
pip install -e .
```

## Usage

Look at the [documentation](https://pixano.github.io/pixano-inference/latest/) to use Pixano-Inference.

## License

Pixano-Inference is released under the terms of the [CeCILL-C license](LICENSE).
