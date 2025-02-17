from fire import Fire

from dmbert.config import BaseModels
from dmbert.embed_distill import EmbedDistill


def main(
    base_model_name: str,
    pca_dims: int,
    model_repository_name: str,
    save_path: str = None,
    commit_msg: str = None,
) -> None:
    """
    Distill the embeddings of a base model to a smaller dimension using Model2Vec framework save the distilled model
    locally and push it to the Hugging Face Hub.

    Args:
        base_model_name (str): Name of the base model to distill. Choose from ['MODERNBERT_BASE', 'MODERNBERT_LARGE',
            'MULTILINGUAL_MPNET_BASE', 'MULTILINGUAL_MINILM_L12'].
        pca_dims (int): Number of dimensions to reduce the embeddings to.
        model_repository_name (str): Name of the repository to push the distilled model to.
        save_path (str, optional): Path to save the distilled model. Defaults to None.
        commit_msg (str, optional): Commit message for the push to the repository. Defaults to None.

    Raises:
        ValueError: If the base model name is not valid.
    """
    if base_model_name not in BaseModels.__members__:
        raise ValueError(f"Invalid base model name. Choose from {BaseModels.__members__.keys()}")

    base_model_name = BaseModels[base_model_name].value

    embed_distill = EmbedDistill(base_model_name, pca_dims, save_path)
    distilled_model = embed_distill.distill()
    embed_distill.push_to_hg_hub(distilled_model, model_repository_name, commit_msg)


if __name__ == "__main__":
    Fire(main)
