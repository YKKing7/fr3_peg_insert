"""Randomly mix robomimic-style HDF5 demos into one dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py


def _demo_sort_key(name: str) -> tuple[int, int | str]:
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return (0, int(suffix))
    return (1, name)


def mix_demos(inputs: list[Path], output: Path, seed: int, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    entries: list[tuple[Path, str]] = []
    totals: list[int] = []
    data_attrs = None
    root_attrs = None

    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
        with h5py.File(input_path, "r") as src:
            if "data" not in src:
                raise KeyError(f"Input dataset has no /data group: {input_path}")
            demos = sorted(src["data"].keys(), key=_demo_sort_key)
            entries.extend((input_path, demo) for demo in demos)
            totals.append(int(src["data"].attrs.get("total", 0)))
            current_data_attrs = dict(src["data"].attrs)
            current_data_attrs.pop("total", None)
            if data_attrs is None:
                data_attrs = current_data_attrs
                root_attrs = dict(src.attrs)
            elif current_data_attrs != data_attrs:
                raise ValueError(f"/data attributes mismatch in {input_path}")

    rng = random.Random(seed)
    rng.shuffle(entries)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as dst:
        if root_attrs is not None:
            for key, value in root_attrs.items():
                dst.attrs[key] = value
        data_dst = dst.create_group("data")
        if data_attrs is not None:
            for key, value in data_attrs.items():
                data_dst.attrs[key] = value
        data_dst.attrs["total"] = sum(totals)

        open_sources: dict[Path, h5py.File] = {}
        try:
            for out_index, (input_path, demo_name) in enumerate(entries):
                src = open_sources.get(input_path)
                if src is None:
                    src = h5py.File(input_path, "r")
                    open_sources[input_path] = src
                src.copy(src["data"][demo_name], data_dst, name=f"demo_{out_index}")
        finally:
            for src in open_sources.values():
                src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Input HDF5 datasets.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output HDF5 dataset.")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for demo order.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    args = parser.parse_args()

    mix_demos(args.inputs, args.output, args.seed, args.overwrite)
    print(f"Wrote mixed dataset: {args.output}")


if __name__ == "__main__":
    main()
