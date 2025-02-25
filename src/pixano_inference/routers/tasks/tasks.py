# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for Image tasks."""

from fastapi import APIRouter

from .image import router as image_router
from .multimodal import router as multimodal_router
from .nlp import router as nlp_router
from .video import router as video_router


router = APIRouter(prefix="/tasks", tags=["Tasks"])
router.include_router(image_router)
router.include_router(multimodal_router)
router.include_router(nlp_router)
router.include_router(video_router)
