from pathlib import Path
import pytest
import torch
from brats_debate.config import load_config
from brats_debate.data.synthetic import create_synthetic


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    path = create_synthetic(tmp_path, Path(__file__).parents[1] / "configs" / "brats.yaml", patients=4)
    return load_config(path)
