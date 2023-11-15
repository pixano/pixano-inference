# Contributing to Pixano Inference

Thank you for your interest in Pixano Inference! Here you will find information on running Pixano Inference locally and guidelines on how to publish your contributions.

## Getting started

### Issue and suggestions

If you find a bug or you think of some missing features that could be useful while using Pixano Inference, please [open an issue](https://github.com/pixano/pixano-inference/issues)!

### Modifications

To contribute more actively to the project, you are welcome to develop the fix or the feature you have in mind, and [create a pull request](https://github.com/pixano/pixano-inference/pulls)!

And if you want to change the module to your liking, feel free to [fork this repository](https://github.com/pixano/pixano-inference/fork)!

## Running Pixano Inference locally

If you are looking to contribute to Pixano Inference and develop new features, you will need to clone the Pixano Inference repository and run it locally.

### Requirements

You will need `python == 3.10`. Then, inside the root `pixano_inference/` directory, run this command to install all the Python dependencies:

```bash
pip install .
```

## Formatting the code

We use these extensions for formatting the Pixano Inference source code:

- Black: Python, Jupyter
  - https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter
- Prettier: Typescript, Javascript, Svelte, HTML, CSS, JSON, YAML, Markdown
  - https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode
  - https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode

## Updating the changelog

When you want to create a pull request with the changes you have made, please update the CHANGELOG.md accordingly.
