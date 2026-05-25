#!/usr/bin/env python3

"""Subset an AnnData object down to cells whose label contains "normal".

The preserved README notes that the missing normal.h5ad file was produced by a
script called am.py from a larger dataset. This utility recreates that behavior
in a configurable form.
"""

import argparse
from typing import Optional, Sequence

import scanpy as sc


def _infer_obs_key(adata, requested: Optional[str] = None) -> str:
    if requested and requested in adata.obs.columns:
        return requested

    candidates = ["celltype", "cell_type", "annotation", "annot", "label", "cell_label"]
    for candidate in candidates:
        if candidate in adata.obs.columns:
            return candidate

    if adata.obs.shape[1] == 1:
        return adata.obs.columns[0]

    raise ValueError("Could not infer the label column. Pass --obs-key explicitly.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Subset an AnnData object to normal cells.")
    parser.add_argument("--input", required=True, help="Input AnnData file.")
    parser.add_argument("--output", required=True, help="Output AnnData file.")
    parser.add_argument("--obs-key", default=None, help="Observation column containing the labels.")
    parser.add_argument(
        "--match",
        default="normal",
        help="Case-insensitive substring used to select normal cells.",
    )
    args = parser.parse_args(argv)

    adata = sc.read_h5ad(args.input)
    obs_key = _infer_obs_key(adata, args.obs_key)
    labels = adata.obs[obs_key].astype(str)
    mask = labels.str.contains(args.match, case=False, na=False)
    subset = adata[mask].copy()
    subset.write_h5ad(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())