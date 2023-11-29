# PyTorch Mediator

A template repository that showcases a streamlined approach to wrapping PyTorch models from research repositories into easy to use objects with clear, typed APIs.

## Structure

- `model.py`: wraps the original model as a PyTorch module with an enhanced API
- `manager.py`: wraps the original model with an enhanced inference-only API
- `checkpoint.py`: provides easy reference to checkpoints
- `checkpoints/`: checkpoint files[^1]
- `port/`: files from the original repository after tweaks such as import renaming[^2]
- `.source/`: unchanged files of the original repository[^3]

[^1]: You will probably want to use DVC or similar tools for large checkpoints
[^2]: My suggestion is to only fix what is strictly needed, i.e. imports, bugs, and possibly unoptimized code
[^3]: Having read-only source files always at hand is useful when we want to verify what changed compared to the original code via tools like `diff`

## API Showcase

### Model

As it is (understandably) often the case with research code, the `forward` method of the toy model to wrap has a rather obscure signature:

```python
class SourceModel:

    def forward(self, x, y, a=1, b=0.0):
        ...
```

Instead, the `Model` wrapper from PyTorch Mediator has a well documented API that is easy to use:

```python
class Model(torch.nn.Module):

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
        ...
```

The classes `Model.OptionalArgs` and `Model.Output` appearing in the signature are also appropriately documented.

### Manager

The `Manager` wrapper also provides a nice, typed API for inference. Moreover, this class also provides a convenient factory method for easy instantiation:

```python
class Manager:

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
        ...
```
