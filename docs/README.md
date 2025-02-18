<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Documentation of Pixano-Inference

## Mkdocs

Pixano-Inference documentation relies on [MkDocs](https://www.mkdocs.org/) which offers a great set of features to generate documentations using Markdown.

## Locally serve

To serve locally the doc you need to install the dependencies:

```bash
cd pixano-inference

pip install ".[docs]"
```

Then you can serve it using MkDocs:

```bash
mkdocs serve -a localhost:8000
```

It will listen to modifications in the `docs/` folder and the `mkdocs.yml` configuration file.
