# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for NLP tasks."""

from fastapi import APIRouter


router = APIRouter(prefix="/nlp", tags=["NLP"])
