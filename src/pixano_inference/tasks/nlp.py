# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Natural Language Processing tasks."""

from .task import Task


class NLPTask(Task):
    """Natural Language Processing tasks."""

    CAUSAL_LM = "causal_lm"
    CONDITONAL_GENERATION = "text_conditional_generation"
    MASKED_LM = "masked_lm"
    MASK_GENERATION = "text_mask_generation"
    MULTIPLE_CHOICE = "multiple_choice"
    NEXT_SENTENCE_PREDICTION = "next_sentence_prediction"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TEXT_ENCODING = "text_encoding"
    TOKEN_CLASSIFICATION = "token_classification"
