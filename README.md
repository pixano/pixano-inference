<div align="center">

<img src="https://raw.githubusercontent.com/pixano/pixano-inference/main/docs/assets/pixano_wide.png" alt="Pixano" height="100"/>

<br/>

**Inference models for Pixano**

**_Under active development, subject to API change_**

[![GitHub version](https://img.shields.io/github/v/release/pixano/pixano-inference?label=release&logo=github)](https://github.com/pixano/pixano-inference/releases)
[![Documentation](https://img.shields.io/website/https/pixano.github.io?up_message=online&up_color=green&down_message=offline&down_color=orange&label=docs)](https://pixano.github.io/pixano-inference)
[![License](https://img.shields.io/badge/license-CeCILL--C-green.svg)](LICENSE)
[![Python version](https://img.shields.io/pypi/pyversions/pixano?color=important&logo=python&logoColor=white)](https://www.python.org/downloads/)

</div>

<hr />

<a href="https://github.com/pixano/pixano" target="_blank">**Pixano**</a> is an open-source tool by CEA List for exploring and annotating your dataset using AI features like **smart segmentation** and **semantic search**.

**Pixano Inference** provides the AI models like _SAM_ and _CLIP_ that power those features, as well as a PyTorch and TensorFlow models for pre-annotating your datasets.

# Getting started

## Installing Pixano Inference

As Pixano and Pixano Inference require specific versions for their dependencies, we recommend creating a new Python virtual environment to install them.

For example, with <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html" target="_blank">conda</a>:

```shell
conda create -n pixano_env python=3.10
conda activate pixano_env
```

Then, you can install Pixano Inference inside that environment with pip:

```shell
pip install pixano-inference
```

As it is a requirement of Pixano Inference, Pixano will automatically be downloaded if it is not already installed.

## Using the models

Please refer to <a href="https://github.com/pixano/pixano/tree/main/notebooks/models" target="_blank">these notebooks</a> for information on how to use the models provided by Pixano Inference.

# Contributing

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for information on running Pixano locally and guidelines on how to publish your contributions.

# License

Pixano Inference is licensed under the [CeCILL-C license](LICENSE).
