#!/usr/bin/env python3

"""Rebuild the affinity-matrix outputs from AnnData inputs and gene-set libraries.

The original repository did not contain the source scripts, only the generated
CSVs and figures. This script reconstructs the observed pipeline:

* aggregate expression by cell type into gene-by-celltype matrices
* emit an ENSEMBL-indexed matrix plus a gene-symbol version
* binarize the gene matrix for downstream term enrichment
* build a term-by-celltype matrix from a TF/GO-style library
* compute cosine similarity and distance matrices
* generate PCA/UMAP visualizations and influential-term summaries

The script is intentionally configurable because the exact raw input paths and
the external transcription-factor library were not preserved in the checkout.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:
    umap = None


DEFAULT_SEVEN_CELLTYPES = ["CD8 T", "Mono", "B", "DC", "NK", "other T", "CD4 T"]


def _read_adata(path: Path):
    return sc.read_h5ad(str(path))


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    return str(value)


def _infer_obs_key(adata, requested: Optional[str] = None) -> str:
    if requested and requested in adata.obs.columns:
        return requested

    candidates = [
        "celltype",
        "cell_type",
        "celltype_major",
        "cell_type_major",
        "annotation",
        "annot",
        "cluster",
        "label",
        "cell_label",
    ]
    for candidate in candidates:
        if candidate in adata.obs.columns:
            return candidate

    if adata.obs.shape[1] == 1:
        return adata.obs.columns[0]

    raise ValueError(
        "Could not infer the observation column that stores cell-type labels. "
        "Pass --obs-key explicitly."
    )


def _load_label_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r") as handle:
        mapping = json.load(handle)
    return {str(key): str(value) for key, value in mapping.items()}


def _get_expression_matrix(adata, use_raw: bool = False, layer: Optional[str] = None):
    if use_raw and adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names.astype(str))
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"Layer {layer!r} was not found in the AnnData object.")
        return adata.layers[layer], pd.Index(adata.var_names.astype(str))
    return adata.X, pd.Index(adata.var_names.astype(str))


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _select_celltypes(
    labels: pd.Series,
    requested_order: Optional[Sequence[str]] = None,
) -> List[str]:
    if requested_order:
        requested = [str(item) for item in requested_order]
        present = set(labels.astype(str))
        selected = [item for item in requested if item in present]
        if selected:
            return selected
    if pd.api.types.is_categorical_dtype(labels):
        return [str(item) for item in labels.cat.categories]
    return _ordered_unique(labels.astype(str).tolist())


def _group_mean_matrix(
    adata,
    obs_key: str,
    celltype_order: Optional[Sequence[str]] = None,
    label_map: Optional[Mapping[str, str]] = None,
    use_raw: bool = False,
    layer: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    matrix, gene_names = _get_expression_matrix(adata, use_raw=use_raw, layer=layer)
    labels = adata.obs[obs_key].map(_normalize_text)

    if label_map:
        labels = labels.map(lambda value: label_map.get(value, value))

    labels = pd.Series(labels.values, index=adata.obs_names, name=obs_key)
    labels = labels.fillna("unknown")
    order = _select_celltypes(labels, celltype_order)

    aggregated = []
    used_order = []
    for label in order:
        mask = labels == label
        if not bool(mask.any()):
            continue
        subset = matrix[mask.values]
        if sp.issparse(subset):
            mean_values = np.asarray(subset.mean(axis=0)).ravel()
        else:
            mean_values = np.asarray(subset.mean(axis=0)).ravel()
        aggregated.append(mean_values)
        used_order.append(label)

    if not aggregated:
        raise ValueError("No cell types were selected from the AnnData object.")

    data = np.vstack(aggregated).T
    frame = pd.DataFrame(data, index=gene_names, columns=used_order)
    return frame, labels


def _gene_symbol_index(adata, gene_names: pd.Index) -> pd.Index:
    candidate_columns = [
        "gene_symbols",
        "gene_symbol",
        "gene_name",
        "symbol",
        "hgnc_symbol",
        "feature_name",
    ]
    for column in candidate_columns:
        if column in adata.var.columns:
            values = adata.var[column].map(_normalize_text)
            filled = []
            for idx, value in enumerate(values.astype(str).tolist()):
                if value and value != "nan":
                    filled.append(value)
                else:
                    filled.append(str(gene_names[idx]))
            return pd.Index(filled, name=None)
    return pd.Index(gene_names.astype(str), name=None)


def _deduplicate_index(frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.Index(frame.index.astype(str))
    keep = ~index.duplicated(keep="first")
    return frame.loc[keep].copy()


def _load_term_library(path: Path) -> Dict[str, List[str]]:
    if path.suffix.lower() == ".gmt":
        library: Dict[str, List[str]] = {}
        with path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                term = parts[0]
                genes = [gene for gene in parts[2:] if gene]
                library[term] = genes
        return library

    with path.open("r") as handle:
        raw = json.load(handle)

    library = {}
    if isinstance(raw, dict):
        for term, entry in raw.items():
            if isinstance(entry, dict):
                genes = entry.get("geneSymbols") or entry.get("genes") or entry.get("gene_symbols") or []
            elif isinstance(entry, list):
                genes = entry
            else:
                genes = []
            cleaned = [_normalize_text(gene) for gene in genes if _normalize_text(gene)]
            if cleaned:
                library[str(term)] = _ordered_unique(cleaned)
    return library


def _build_term_matrix(binary_gene_matrix: pd.DataFrame, library: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    gene_index = pd.Index(binary_gene_matrix.index.astype(str))
    gene_lookup = set(gene_index)

    rows = []
    terms = []
    for term in sorted(library.keys()):
        genes = [_normalize_text(gene) for gene in library[term]]
        genes = [gene for gene in _ordered_unique(genes) if gene in gene_lookup]
        if genes:
            values = binary_gene_matrix.loc[genes].mean(axis=0)
        else:
            values = pd.Series(0.0, index=binary_gene_matrix.columns)
        rows.append(values.values)
        terms.append(term)

    return pd.DataFrame(rows, index=terms, columns=binary_gene_matrix.columns)


def _cosine_similarity_frame(matrix: pd.DataFrame, hyphenate_labels: bool = False) -> pd.DataFrame:
    labels = [str(column) for column in matrix.columns]
    if hyphenate_labels:
        labels = [label.replace(" ", "-") for label in labels]

    values = np.asarray(matrix.T.values, dtype=float)
    similarity = cosine_similarity(values)
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return pd.DataFrame(similarity, index=labels, columns=labels)


def _cosine_distance_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    labels = [str(column) for column in matrix.columns]
    values = np.asarray(matrix.T.values, dtype=float)
    similarity = cosine_similarity(values)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=labels, columns=labels)


def _save_heatmap(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(frame, cmap="viridis", square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def _save_scatter(positions: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.set_style("whitegrid")
    plt.scatter(positions.iloc[:, 0], positions.iloc[:, 1], s=75)
    for label, row in positions.iterrows():
        plt.text(row.iloc[0] + 0.01, row.iloc[1] + 0.01, label, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def _pca_embedding(distance_frame: pd.DataFrame) -> pd.DataFrame:
    values = StandardScaler().fit_transform(distance_frame.values)
    n_components = min(2, values.shape[0], values.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(values)
    columns = ["PC1", "PC2"][:n_components]
    return pd.DataFrame(coords, index=distance_frame.index, columns=columns)


def _umap_embedding(distance_frame: pd.DataFrame) -> pd.DataFrame:
    if umap is None:
        values = StandardScaler().fit_transform(distance_frame.values)
        coords = PCA(n_components=2, random_state=0).fit_transform(values)
        return pd.DataFrame(coords, index=distance_frame.index, columns=["UMAP1", "UMAP2"])

    n_samples = distance_frame.shape[0]
    neighbors = max(2, min(5, n_samples - 1))
    reducer = umap.UMAP(metric="precomputed", random_state=0, n_neighbors=neighbors, min_dist=0.25)
    coords = reducer.fit_transform(distance_frame.values)
    return pd.DataFrame(coords, index=distance_frame.index, columns=["UMAP1", "UMAP2"])


def _save_elbow_and_loadings(function_matrix: pd.DataFrame, output_dir: Path) -> None:
    feature_matrix = StandardScaler().fit_transform(function_matrix.T.values)
    max_components = min(feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=max_components, random_state=0)
    pca.fit(feature_matrix)

    plt.figure(figsize=(7, 4))
    x_values = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    plt.plot(x_values, pca.explained_variance_ratio_, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Elbow plot for function matrix")
    plt.tight_layout()
    plt.savefig(str(output_dir / "ELBOW_function.png"), dpi=200)
    plt.close()

    loading_series = pd.Series(pca.components_[0], index=function_matrix.index.astype(str))
    ranked = loading_series.reindex(loading_series.abs().sort_values(ascending=False).index)
    top = ranked.head(20).rename("Score").reset_index().rename(columns={"index": "GO_Term"})
    top.to_csv(str(output_dir / "top_influential.tsv"), sep="\t", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score", y="GO_Term", data=top, orient="h")
    plt.title("Top PC1 loadings")
    plt.tight_layout()
    plt.savefig(str(output_dir / "PC1_influential_loadings.png"), dpi=200)
    plt.close()


def _build_similarity_products(function_matrix: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    similarity = _cosine_similarity_frame(function_matrix, hyphenate_labels=True)
    similarity.to_csv(str(output_dir / f"{prefix}_similarity_matrix_GO.csv"))
    similarity.to_csv(str(output_dir / f"{prefix}_similarity_matrix_TF.csv"))
    _save_heatmap(similarity, output_dir / f"{prefix}_similarity_heatmap.png", f"{prefix} cosine similarity")


def _build_distance_products(gene_matrix: pd.DataFrame, output_dir: Path, prefix: str) -> pd.DataFrame:
    distance = _cosine_distance_frame(gene_matrix)
    distance.to_csv(str(output_dir / f"{prefix}_distance_matrix.csv"))
    _save_heatmap(distance, output_dir / f"{prefix}_distance_heatmap.png", f"{prefix} cosine distance")
    return distance


def _rebuild_outputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = _read_adata(Path(args.adata))
    obs_key = _infer_obs_key(adata, args.obs_key)
    label_map = _load_label_map(Path(args.label_map)) if args.label_map else {}

    requested_order = None
    if args.celltype_order:
        requested_order = [item for item in args.celltype_order.split(",") if item]
    elif len(_ordered_unique(adata.obs[obs_key].map(_normalize_text).tolist())) == len(DEFAULT_SEVEN_CELLTYPES):
        requested_order = DEFAULT_SEVEN_CELLTYPES

    gene_matrix, labels = _group_mean_matrix(
        adata,
        obs_key=obs_key,
        celltype_order=requested_order,
        label_map=label_map,
        use_raw=args.use_raw,
        layer=args.layer,
    )
    gene_matrix.to_csv(str(output_dir / "celltype_gene_matrix.csv"))

    symbol_index = _gene_symbol_index(adata, gene_matrix.index)
    symbol_gene_matrix = gene_matrix.copy()
    symbol_gene_matrix.index = symbol_index.values
    symbol_gene_matrix = _deduplicate_index(symbol_gene_matrix)
    symbol_gene_matrix.to_csv(str(output_dir / "celltype_gene_matrix_geneNames.csv"))

    binary_tmp = (symbol_gene_matrix > args.binary_threshold).astype(int)
    binary_tmp.to_csv(str(output_dir / "celltype_gene_matrix_binarized_TMP.csv"))
    binary_tmp.to_csv(str(output_dir / "celltype_gene_matrix_binarized.csv"))

    distance = _build_distance_products(gene_matrix, output_dir, "celltype")

    if args.term_library:
        library = _load_term_library(Path(args.term_library))
        function_matrix = _build_term_matrix(binary_tmp, library)
        function_matrix.to_csv(str(output_dir / "celltype_function_matrix.csv"))
        _build_similarity_products(function_matrix, output_dir, "celltype")
        _save_elbow_and_loadings(function_matrix, output_dir)
    else:
        print("No term library supplied, skipping function matrix and similarity outputs.")

    pca_coords = _pca_embedding(distance)
    _save_scatter(pca_coords, output_dir / "PCAofCosineDistanceMatrix.png", "PCA of cosine distance matrix")

    umap_coords = _umap_embedding(distance)
    _save_scatter(umap_coords, output_dir / "UMAPofCosineDistanceMatrix.png", "UMAP of cosine distance matrix")

    if args.save_grouped_obs:
        labels.rename("celltype").to_frame().to_csv(str(output_dir / f"{args.save_grouped_obs}.csv"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the affinity-matrix outputs.")
    parser.add_argument("--adata", required=True, help="Path to the AnnData input file.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated outputs.")
    parser.add_argument("--obs-key", default=None, help="Observation column containing cell-type labels.")
    parser.add_argument("--label-map", default=None, help="Optional JSON map from fine labels to grouped labels.")
    parser.add_argument(
        "--celltype-order",
        default=None,
        help="Comma-separated cell-type order to preserve in the output matrices.",
    )
    parser.add_argument(
        "--term-library",
        default=None,
        help="Optional GO/TFT style library in JSON or GMT format.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.0,
        help="Threshold applied when binarizing the gene matrix.",
    )
    parser.add_argument("--layer", default=None, help="AnnData layer to use instead of X.")
    parser.add_argument("--use-raw", action="store_true", help="Use adata.raw when available.")
    parser.add_argument(
        "--save-grouped-obs",
        default=None,
        help="If set, write the resolved cell-type labels to this basename in the output directory.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _rebuild_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())