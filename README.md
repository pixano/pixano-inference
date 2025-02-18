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

This library aims to provide a common ecosystem to launch inference for various Artificial Intelligence tasks from different providers (Open-AI, transformers, sam2, ...). It has first been implemented to work in par with the [Pixano](https://pixano.github.io/pixano/latest/) AI-powered annotation tool.

## Installation

To install the library, simply execute the following command

```bash
pip install pixano-inference
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
