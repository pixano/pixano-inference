# @Copyright: CEA-LIST/DIASI/SIALV/LVA (2023)
# @Author: CEA-LIST/DIASI/SIALV/LVA <pixano@cea.fr>
# @License: CECILL-C
#
# This software is a collaborative computer program whose purpose is to
# generate and explore labeled data for computer vision applications.
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
#
# http://www.cecill.info

import importlib
from types import ModuleType


def attempt_import(module: str, package: str = None) -> ModuleType:
    """Import specified module, or raise ImportError with a helpful message

    Args:
        module (str): The name of the module to import
        package (str): The package to install, None if identical to module name. Defaults to None.

    Returns:
        ModuleType: Imported module
    """

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"Please install {module.split('.')[0]} to use this model: pip install {package or module}"
        ) from e
