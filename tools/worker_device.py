"""Which device each tokenisation or decoding worker runs on.

`tokenize_audio` and `decode_tokens` were written for a multi-GPU box and hardcoded
`torch.device("cuda", rank)`. That forces every wav through a rented instance, which is the
most expensive way to run the cheapest part of the pipeline: Mimi encode/decode is small
enough to run on a laptop, and audio that never leaves this machine cannot leak from a
rented one either.

The rules live here as plain functions so they can be tested without torch, and so the two
scripts cannot drift apart on what `--device mps` is supposed to mean.
"""

from __future__ import annotations


class NoAcceleratorError(RuntimeError):
    """Raised when CUDA was asked for and no GPU is visible."""


def resolve_worker_device(device_spec: str, worker_index: int) -> str:
    """The device string for one worker.

    Plain "cuda" fans out, one worker per GPU, which is what the original code did. Every
    other spec names a single device that all workers share.
    """
    if device_spec == "cuda":
        return f"cuda:{worker_index}"
    return device_spec


def resolve_worker_count(requested: int, device_spec: str, *, available_cuda: int) -> int:
    """How many workers can actually run on `device_spec`.

    Falling back to CPU when CUDA was requested is never the kind thing to do here: on a
    billing instance it turns a two-minute job into an overnight one without saying so. So
    a missing GPU raises instead, and CPU has to be asked for by name.
    """
    if requested < 1:
        raise ValueError(f"num_workers must be at least 1, got {requested}")

    if device_spec == "cuda":
        if available_cuda < 1:
            raise NoAcceleratorError(
                "--device cuda was requested but no CUDA device is visible. "
                "Pass --device cpu or --device mps to run without a GPU."
            )
        return min(requested, available_cuda)

    # One named device cannot be divided between processes: mps exposes a single GPU with
    # no multi-process story, and an explicit cuda:N means that card and no other.
    if device_spec.startswith("mps") or device_spec.startswith("cuda:"):
        return 1

    return requested
