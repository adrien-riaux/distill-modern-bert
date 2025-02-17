import pytest
from pytest_mock import MockerFixture

from dmbert.config import BaseModels
from dmbert.embed_distill import EmbedDistill


@pytest.fixture
def mock_torch_cuda(mocker: MockerFixture) -> None:
    """
    Fixture to mock the torch.cuda.is_available function.

    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    """
    mocker.patch("torch.cuda.is_available", return_value=False)


@pytest.fixture
def embed_distill_fixture() -> EmbedDistill:
    """
    Fixture to create an EmbedDistill object with ModernBERT Base as the base model and 128 as the PCA dimensions.

    Returns:
        EmbedDistill: EmbedDistill object with ModernBERT Base as the base model and 128 as the PCA dimensions.
    """
    return EmbedDistill(base_model_name=BaseModels.MODERNBERT_BASE, pca_dims=128)
