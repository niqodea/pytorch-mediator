import dataclasses
from typing import Any

import torch

from port.main import SourceModel


class Model(torch.nn.Module):
    """
    Model wrapper for the source model. Provides a nicer interface for the model.
    """

    def __init__(self, source_model: SourceModel):
        """
        :param source_model: Source model to wrap.
        """

        super().__init__()
        self.source_model = source_model

    @dataclasses.dataclass
    class OptionalArgs:
        """
        Optional arguments for the model. If None, the default value is used.
        """

        alpha: int | None = None  # > 0
        """
        Alpha value.
        """
        beta: float | None = None  # >= 0.0
        """
        Beta value.
        """

    @dataclasses.dataclass
    class Output:
        """
        Output of the model.
        """

        baz: torch.Tensor  # B, 3, H, W
        """
        Output image.
        """
        qux: torch.Tensor  # B, 1, H, W
        """
        Output mask.
        """

    def forward(
        self,
        foo: torch.Tensor,  # B, 3, H, W
        bar: torch.Tensor,  # B, 3, H, W
        optional_args: "Model.OptionalArgs" = OptionalArgs(),
    ) -> "Model.Output":
        """
        Run inference with the model.

        :param foo: First input image.
        :param bar: Second input image.
        :param optional_args: Optional arguments.
        :return: Output of the model.
        """

        baz, qux = self.source_model.forward(
            x=foo,
            y=bar,
            **Model._get_optional_kwargs(optional_args),
        )

        return Model.Output(
            baz=baz,
            qux=qux,
        )

    @staticmethod
    def _get_optional_kwargs(optional_args: "Model.OptionalArgs") -> dict[str, Any]:
        return (
            {"a": optional_args.alpha} if optional_args.alpha is not None else {}
        ) | ({"b": optional_args.beta} if optional_args.beta is not None else {})
