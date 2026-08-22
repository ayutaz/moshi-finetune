"""Work out a training run's shape before renting the GPU that would tell you.

`finetune.py` prints Num examples, batch size and total steps only after loading an 8.4B
model onto two GPUs, so the cheapest way to learn them is to compute them - the formulae
are three lines and are mirrored here from finetune.py:736-743.

Two of these numbers decide whether a run survives.

`--save_steps` has to be pinned on the command line before launch, and it is only "once per
epoch" if it equals the realised steps-per-epoch. Get it wrong and checkpoints land off the
epoch boundaries they are supposed to mark.

The checkpoint count decides the disk. `finetune.py` saves in-loop whenever
`current_steps % save_steps == 0` and then saves again unconditionally after the loop
(finetune.py:977-979 and :985-986), with no rotation anywhere - so the count is not simply
the number of epochs. With 72 rows at batch 8, S is 9 and the post-loop save reuses step_45:
five checkpoints. With 80 rows, S is 10, total is 50, the in-loop saves still land on
multiples of 9, and the post-loop save adds step_50 - six checkpoints at 100.46 GB each,
which is 100 GB of disk nobody planned for.
"""

from __future__ import annotations

import math


def global_batch_size(*, per_device: int, processes: int, gradient_accumulation: int) -> int:
    """Examples consumed per optimisation step across the whole run."""
    for name, value in (
        ("per_device", per_device),
        ("processes", processes),
        ("gradient_accumulation", gradient_accumulation),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    return per_device * processes * gradient_accumulation


def steps_per_epoch(*, examples: int, batch: int) -> int:
    """Optimisation steps in one epoch, rounding up for the partial final batch."""
    if examples < 1:
        raise ValueError(f"a training set needs at least one example, got {examples}")
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    return math.ceil(examples / batch)


def total_steps(*, examples: int, batch: int, epochs: int) -> int:
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    return epochs * steps_per_epoch(examples=examples, batch=batch)


def checkpoint_steps(*, examples: int, batch: int, epochs: int, save_steps: int) -> list[int]:
    """Every step at which a checkpoint directory is written.

    Includes the unconditional post-loop save, which only coincides with an in-loop save
    when `save_steps` divides the total.
    """
    if save_steps < 1:
        raise ValueError(f"save_steps must be at least 1, got {save_steps}")
    final = total_steps(examples=examples, batch=batch, epochs=epochs)
    steps = list(range(save_steps, final + 1, save_steps))
    if final not in steps:
        steps.append(final)
    return steps
