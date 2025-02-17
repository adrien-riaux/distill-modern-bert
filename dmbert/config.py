from enum import StrEnum
from enum import unique


@unique
class BaseModels(StrEnum):
    """
    Enum class for the base models available for distillation.
    """

    MODERNBERT_BASE = "nomic-ai/modernbert-embed-base"
    MODERNBERT_LARGE = "lightonai/modernbert-embed-large"
    MULTILINGUAL_MPNET_BASE = "paraphrase-multilingual-mpnet-base-v2"
    MULTILINGUAL_MINILM_L12 = "paraphrase-multilingual-MiniLM-L12-v2"
