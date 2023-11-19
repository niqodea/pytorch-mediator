from typing import TypeAlias

import torch

from checkpoint import Checkpoint
from model import Model
from port.main import SourceModel


class Manager:
    """
    Manager for the model. Provides a high-level interface for running inference.
    """

    def __init__(self, model: Model):
        """
        :param model: Model to manage.
        """
        self._model = model

    OptionalArgs: TypeAlias = Model.OptionalArgs

    Output: TypeAlias = Model.Output

    def infer(
        self,
        foo: torch.Tensor,  # B, 3, H, W
        bar: torch.Tensor,  # B, 3, H, W
        optional_args: "Manager.OptionalArgs" = OptionalArgs(),
    ) -> "Manager.Output":
        """
        Run inference with the model.

        :param foo: First input image.
        :param bar: Second input image.
        :param optional_args: Optional arguments.
        :return: Output image.
        """

        if foo.shape != bar.shape:
            raise ValueError("foo and bar must have the same shape")

        with torch.inference_mode():
            output = self._model.forward(foo, bar, optional_args)

        return output

    @staticmethod
    def create(
        checkpoint: Checkpoint = Checkpoint.v1_1,
        device: torch.device = torch.device("cuda"),
        half: bool = False,
    ) -> "Manager":
        """
        Create a new manager for the model.

        :param checkpoint: Checkpoint to load.
        :param device: Device to load the model on.
        :param half: Whether to use half precision.
        :return: New manager.
        """

        checkpoint_path = checkpoint.value
        if not checkpoint_path.exists():
            raise ValueError(
                f"Checkpoint file for {checkpoint.name} not found at {checkpoint_path}"
            )
        state_dict = torch.load(checkpoint_path)

        source_model = SourceModel()
        source_model.load_state_dict(state_dict)

        source_model.eval()
        source_model = source_model.to(device)
        if half:
            source_model = source_model.half()

        model = Model(source_model)
        model = model.to(device)
        manager = Manager(model)

        return manager
