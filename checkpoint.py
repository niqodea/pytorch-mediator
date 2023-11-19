from enum import Enum
from pathlib import Path

_CHECKPOINTS_PATH = Path(__file__).parent / "checkpoints"


class Checkpoint(Enum):
    """
    Represents a checkpoint for the model and contains its path.
    """

    v1_0 = _CHECKPOINTS_PATH / "v1_0.pth"
    """
    v1.0 checkpoint. It is the first checkpoint that was released.
    Source URL: https://drive.google.com/drive/folders/1234567890
    """
    v1_1 = _CHECKPOINTS_PATH / "v1_1.pth"
    """
    v1.1 checkpoint. It is the second checkpoint that was released.
    Source URL: https://drive.google.com/drive/folders/abcdefghij
    """
