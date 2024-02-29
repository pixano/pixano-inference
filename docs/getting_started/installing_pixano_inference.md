As Pixano and Pixano Inference require specific versions for their dependencies, we recommend creating a new Python virtual environment to install them.

For example, with <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html" target="_blank">conda</a>:

```shell
conda create -n pixano_env python=3.10
conda activate pixano_env
```

Then, you can install the Pixano and Pixano Inference packages inside that environment with pip:

```shell
pip install pixano
pip install pixano-inference
```

To use the inference models available through GitHub, install the following additional packages:

```shell
python -m pip install segment-anything@git+https://github.com/facebookresearch/segment-anything
python -m pip install mobile-sam@git+https://github.com/ChaoningZhang/MobileSAM
```
