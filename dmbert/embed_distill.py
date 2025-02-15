import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding


class EmbedDistill:
    """
    Class to distill a SentenceTransformer model to a smaller model using Model2Vec framework. The
    distilled model is then saved and can be pushed to Hugging Face Hub. If available, the distillation
    process is done on GPU.

    Args:
        base_model_name (str): Name of the base model to distill.
        pca_dims (int): Number of PCA dimensions to reduce the model to.
        save_path (str, optional): Path to save the distilled model. Defaults to None.
    """

    def __init__(self, base_model_name: str, pca_dims: int, save_path: str = None) -> None:
        self.base_model_name = base_model_name
        self.pca_dims = pca_dims

        self._save_path = save_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def save_path(self) -> str:
        """
        Property to get the save path for the distilled model.

        Returns:
            str: Save path for the distilled model.
        """
        model_path = Path(__file__).resolve().parent.parent / "models"
        model_path.mkdir(exist_ok=True)

        if self._save_path is None:
            self._save_path = str(
                model_path / f"distilled-{self.base_model_name.split('/')[-1]}-dim-{self.pca_dims}"
            )

        return self._save_path

    def distill(self) -> SentenceTransformer:
        """
        Distill the base model to a smaller model using Model2Vec framework.

        Returns:
            SentenceTransformer: The distilled model.
        """
        logger.info(f"Distilling base model {self.base_model_name}")
        static_embedding = StaticEmbedding.from_distillation(
            self.base_model_name, pca_dims=self.pca_dims, device=self._device
        )
        distilled_model = SentenceTransformer(modules=[static_embedding])

        logger.info(f"Saving distilled model to {self.save_path}")
        distilled_model.save(self.save_path, safe_serialization=False)

        return distilled_model

    def push_to_hg_hub(
        self,
        distilled_model: SentenceTransformer,
        model_repository_name: str,
        commit_msg: str = None,
    ) -> None:
        """
        Push the distilled model to Hugging Face Hub. The Hub token is read from .env file, and should
        be named as HG_TOKEN.

        Args:
            distilled_model (SentenceTransformer): The distilled model to push to Hugging Face Hub.
            model_repository_name (str): Name of the repository to push the model to.
            commit_msg (str, optional): Commit message for the push. Defaults to None.

        Raises:
            ValueError: If Hugging Face token is not found.
        """
        load_dotenv()
        token = os.getenv("HG_TOKEN")

        if not token:
            logger.error("Hugging Face token not found.")
            raise ValueError("Hugging Face token not found. Please set HG_TOKEN in .env file.")

        logger.info("Pushing distilled model to Hugging Face Hub")
        distilled_model.push_to_hub(
            model_repository_name,
            token=token,
            commit_message=commit_msg,
            exist_ok=True,
            safe_serialization=False,
            replace_model_card=False,
        )
