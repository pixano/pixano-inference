<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Pixano Inference API reference

## Client module

The client module contains the class for the API client. It is responsible for making requests to the API endpoints.

## Data module

The data module contains the functions to read (and later write) data from/to a database or file.

## Model registry module

The model registry module contains the functions to register a model to the application.

## Models module

The models module contains the inference models to perform the tasks Pixano Inference API is designed for.

The models include:

- `BaseInferenceModel`: Base class for all Pixano Inference API models.
- `Sam2Model`: Model used to detect and segment objects in images and videos.
- `TransformerModel`: Model instantiated from Transformers.
- `VLLMModel`: Model instantiated from VLLM.

## Providers module

The providers module contains the functions to load a model and to perform inference either from a model provider or from an API provider.

The providers include:

- `BaseProvider`: Base class for all Pixano Inference API providers.
- `Sam2Provider`: Provider used to instantiate a `Sam2Model` and call its methods.
- `TransformersProvider`: Provider used to instantiate a `TransformerModel` and call its methods.
- `VLLMProvider`: Provider used to instantiate a `VLLMModel` and call its methods.

## Pydantic module

The pydantic module contains the classes for data validation. It is used by the models, providers, and the application itself to validate the input/output of the API.

## Routers module

The routers module contains the routers for the API. Each router has a path prefix that defines the endpoint where it will be mounted in the API.

The routers swagger is accessible at at the `/docs` endpoint.

## Settings module

The settings module contains the configuration of the application.

## Tasks module

The tasks module contains the enums used to define the task that a model can perform.

## Utils module

The utils module contains the functions and classes used by the other modules.
