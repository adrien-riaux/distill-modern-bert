from enum import StrEnum
from enum import unique


@unique
class BaseModels(StrEnum):
    """
    Enum class for the base models available for distillation.
    """

    MODERNBERT_BASE = "nomic-ai/modernbert-embed-base"
    MODERNBERT_LARGE = "lightonai/modernbert-embed-large"
