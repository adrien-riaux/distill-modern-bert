from pathlib import Path

from dmbert.config import BaseModels
from dmbert.embed_distill import EmbedDistill


def test_initialization(mock_torch_cuda: None, embed_distill_fixture: EmbedDistill) -> None:
    """
    Test the initialization of the EmbedDistill object. It uses the embed distill fixture to get the EmbedDistill
    object.

    Args:
        mock_torch_cuda (None): Mock torch cuda availability.
        embed_distill_fixture (EmbedDistill): EmbedDistill fixture.

    Raises:
        AssertionError: If any assertion fails.
    """
    assert embed_distill_fixture.base_model_name == BaseModels.MODERNBERT_BASE
    assert embed_distill_fixture.pca_dims == 128
    assert embed_distill_fixture._device == "cpu"


def test_save_path(embed_distill_fixture: EmbedDistill) -> None:
    """
    Test the save path for the distilled model. It uses the embed distill fixture to get the save path.

    Args:
        embed_distill_fixture (EmbedDistill): EmbedDistill fixture.

    Raises:
        AssertionError: If any assertion fails.
    """
    expectec_save_path = Path(embed_distill_fixture.save_path)

    assert expectec_save_path.parent.exists()
    assert expectec_save_path.parent.is_dir()
    assert expectec_save_path.name == "distilled-modernbert-embed-base-dim-128"
    assert expectec_save_path.parent.name == "models"
