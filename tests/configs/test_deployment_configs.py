# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The deployment config classes belong to ``configs`` and stay importable from ``ray.config``."""

import pytest

from pixano_inference import configs
from pixano_inference.configs import deployment
from pixano_inference.ray import config as ray_config


@pytest.mark.parametrize("name", ["ResourceConfig", "AutoscalingConfig", "ModelDeploymentConfig"])
def test_deployment_configs_have_one_definition(name):
    defined = getattr(deployment, name)
    assert defined.__module__ == "pixano_inference.configs.deployment"
    assert getattr(configs, name) is defined
    assert getattr(ray_config, name) is defined
