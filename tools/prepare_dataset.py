import argparse
import os


def merge_text_audio(text_ids, audio_ids, text_padding_id: int):
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [T_text]
        audio_ids: Tokenized audio stream. Shape: [K=8, T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=8+1, T_audio]
    """
    import numpy as np

    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    audio_len = audio_ids.shape[-1]
    if text_ids.shape[0] > audio_len:
        text_ids = text_ids[:audio_len]
    elif text_ids.shape[0] < audio_len:
        text_ids = np.concat(
            [text_ids, np.full(audio_len - text_ids.shape[0], text_padding_id)], axis=0
        )
    return np.concat([text_ids[None], audio_ids], axis=0).astype(np.int32).tolist()


class StemMismatchError(RuntimeError):
    """The tokenized text and audio directories do not describe the same dialogues."""


def matched_dialogue_stems(text_stems: list[str], audio_stems: list[str]) -> list[str]:
    """The dialogue names present in both directories, sorted.

    Raises instead of returning a partial set. The previous behaviour printed a message and
    returned from main(), which exits 0: a shell chaining this with `&&` would carry on,
    the parquet would simply not exist, and the failure would surface much later as a glob
    matching nothing - by which time a GPU is billing.

    Sorted rather than in os.listdir order, because that order decides which dialogue lands
    in which parquet batch and is not stable across machines.
    """
    missing_text = sorted(set(audio_stems) - set(text_stems))
    missing_audio = sorted(set(text_stems) - set(audio_stems))
    if missing_text or missing_audio:
        parts = []
        if missing_text:
            parts.append(f"no tokenized text for {len(missing_text)}: {missing_text[:10]}")
        if missing_audio:
            parts.append(f"no tokenized audio for {len(missing_audio)}: {missing_audio[:10]}")
        raise StemMismatchError("; ".join(parts))

    stems = sorted(set(text_stems))
    if not stems:
        # Two empty directories match perfectly, and would otherwise produce a zero-row
        # parquet and a clean exit.
        raise StemMismatchError("no tokenized dialogues found in either directory")
    return stems


def dialogue_id_for(output_prefix: str, dialogue_name: str) -> str:
    """A dataset-unique id for one dialogue.

    Namespaced by the split rather than by the output path. Upstream joined the whole
    --output_prefix, which bakes the builder's absolute local directories into the dataset:
    a parquet built on a laptop and uploaded to a rented instance carried the laptop's
    paths, and the id could not be joined back to a manifest row. The basename keeps train
    and dev from colliding, which was the point of including the prefix at all.
    """
    return f"{os.path.basename(output_prefix)}/{dialogue_name}"


def main(args):
    import numpy as np  # noqa: F401  (used by merge_text_audio via the loaded npz arrays)
    import pandas as pd
    from tqdm import tqdm

    text_dialogue_names = [os.path.splitext(f)[0] for f in os.listdir(args.tokenized_text_dir)]
    audio_dialogue_names = [os.path.splitext(f)[0] for f in os.listdir(args.tokenized_audio_dir)]
    dialogue_names = matched_dialogue_stems(text_dialogue_names, audio_dialogue_names)

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    num_dialogues = len(dialogue_names)
    num_parquets = -(-num_dialogues // args.num_examples_per_parquet)

    for i in range(num_parquets):
        dials_per_parquet = dialogue_names[
            i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
        ]

        # load the tokenized text and audio data
        data = []
        for dialogue_name in tqdm(
            dials_per_parquet, desc=f"Processing parquet {i + 1}/{num_parquets}"
        ):
            text_path = os.path.join(args.tokenized_text_dir, f"{dialogue_name}.npz")
            text_ids = np.load(text_path)
            audio_path = os.path.join(args.tokenized_audio_dir, f"{dialogue_name}.npz")
            audio_ids = np.load(audio_path)
            data.append(
                {
                    "dialogue_id": dialogue_id_for(args.output_prefix, dialogue_name),
                    "A": merge_text_audio(text_ids["A"], audio_ids["A"], args.text_padding_id),
                    "B": merge_text_audio(text_ids["B"], audio_ids["B"], args.text_padding_id),
                }
            )

        # save the merged data
        df = pd.DataFrame(data)
        output_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the tokenized text and audio data into a single dataset in parquet format."
    )
    parser.add_argument(
        "--tokenized_text_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized text data.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized audio data.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for the output dataset. Output files will be named as "
            "`{{output_prefix}}-001-of-002.parquet` etc."
        ),
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding id for text stream to fill the gap between audio and text streams.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of samples per parquet file.",
    )
    args = parser.parse_args()

    try:
        main(args)
    except StemMismatchError as error:
        raise SystemExit(f"prepare_dataset: {error}") from error
