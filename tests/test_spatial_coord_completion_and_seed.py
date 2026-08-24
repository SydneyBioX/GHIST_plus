#!/usr/bin/env python3
"""CPU tests for coordinate completion/cache and matched-run seeding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataio.spatial as spatial  # noqa: E402


def test_csv_preserved_missing_filled_and_cached_once():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2, 3], "id_xenium": [11, 12, 13]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [11, 12],
                "x_coord": [101.25, 202.5],
                "y_coord": [10.5, 20.75],
            }
        ).to_csv(fp_coords, index=False)
        fp_seg.write_bytes(b"test-placeholder")
        src = SimpleNamespace(
            slide_idx=99,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="xenium",
        )

        calls = []
        original = spatial.centroids_from_label_image

        def fake_centroids(path, cell_ids, chunk_rows=256):
            calls.append((str(path), np.asarray(cell_ids).copy(), int(chunk_rows)))
            assert np.array_equal(np.asarray(cell_ids), np.array([3]))
            return np.array([3]), np.array([[30.0, 303.0]], dtype=np.float32)

        spatial.clear_histology_coord_caches()
        spatial.centroids_from_label_image = fake_centroids
        try:
            first = spatial.load_histology_coord_map_from_source(src)
            second = spatial.load_histology_coord_map_from_source(src)
        finally:
            spatial.centroids_from_label_image = original

        assert len(calls) == 1, "path cache must prevent a second TIFF scan"
        assert first == second
        assert first[1] == (10.5, 101.25), "CSV coordinate was modified"
        assert first[2] == (20.75, 202.5), "CSV coordinate was modified"
        assert first[3] == (30.0, 303.0), "missing coordinate was not centroid-filled"
        stats = spatial.get_histology_coord_completion_stats(src)
        assert stats["cell_coords_id_space"] == "xenium"
        assert stats["cell_coords_id_space_requested"] == "xenium"
        assert stats["matched_histology_ids"] == 3
        assert stats["csv_coordinates"] == 2
        assert stats["csv_coordinate_coverage"] == 2 / 3
        assert stats["centroid_filled"] == 1
        assert stats["unresolved"] == 0
        assert stats["total_coordinates"] == 3
        assert stats["cache_hit"] is True


def test_completion_cache_isolated_by_missing_id_set():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_completion_key_test_") as tmp:
        base = Path(tmp)
        fp_seg = base / "shared_labels.tif"
        fp_seg.write_bytes(b"test-placeholder")
        sources = []
        for name, histology_ids, xenium_ids in (
            ("first", [1, 2], [11, 12]),
            ("second", [1, 2, 3], [11, 12, 13]),
        ):
            source_dir = base / name
            source_dir.mkdir()
            fp_match = source_dir / "matched_nuclei_filtered.csv"
            fp_coords = source_dir / "cell_coords.csv"
            pd.DataFrame(
                {"id_histology": histology_ids, "id_xenium": xenium_ids}
            ).to_csv(fp_match, index=False)
            pd.DataFrame(
                {"cell_id": [11], "x_coord": [101.0], "y_coord": [10.0]}
            ).to_csv(fp_coords, index=False)
            sources.append(
                SimpleNamespace(
                    slide_idx=100 + len(sources),
                    fp_nuc_sizes=str(fp_match),
                    fp_nuc_seg=str(fp_seg),
                    cell_coords_id_space="xenium",
                )
            )

        calls = []
        original = spatial.centroids_from_label_image

        def fake_centroids(path, cell_ids, chunk_rows=256):
            ids = np.asarray(cell_ids, dtype=np.int64)
            calls.append(ids.copy())
            coords = np.stack([ids * 10.0, ids * 100.0], axis=1).astype(np.float32)
            return ids, coords

        spatial.clear_histology_coord_caches()
        spatial.centroids_from_label_image = fake_centroids
        try:
            first = spatial.load_histology_coord_map_from_source(sources[0])
            second = spatial.load_histology_coord_map_from_source(sources[1])
        finally:
            spatial.centroids_from_label_image = original

        assert np.array_equal(calls[0], np.array([2]))
        assert np.array_equal(calls[1], np.array([2, 3]))
        assert first[2] == (20.0, 200.0)
        assert second[3] == (30.0, 300.0)


def test_nonnumeric_coordinate_id_cannot_match_missing_id():
    matched = pd.DataFrame(
        {"id_histology": [1, 2], "id_xenium": [11, np.nan]}
    )
    coordinates = pd.DataFrame(
        {
            "cell_id": [11, "not-an-id"],
            "x_coord": [1.5, 99.0],
            "y_coord": [1.5, 88.0],
        }
    )
    coordinate_map = spatial._coordinate_map_from_table(
        matched,
        coordinates,
        id_column="cell_id",
        x_column="x_coord",
        y_column="y_coord",
        id_space="xenium",
    )
    assert coordinate_map == {1: (1.5, 1.5)}


def _write_tiny_segmentation(path):
    import tifffile

    labels = np.zeros((16, 16), dtype=np.uint32)
    labels[1:3, 1:3] = 1
    labels[10:12, 10:12] = 2
    tifffile.imwrite(path, labels)


def test_histology_namespace_direct_keying_alignment_and_cache_isolation():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_namespace_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [1, 2],
                "x_coord": [1.5, 10.5],
                "y_coord": [1.5, 10.5],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)

        src_histology_unvalidated = SimpleNamespace(
            slide_idx=1,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="histology",
        )
        src_histology = SimpleNamespace(
            slide_idx=1,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="histology",
            cell_coords_validate_alignment=True,
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=1e-6,
        )
        src_auto = SimpleNamespace(
            slide_idx=1,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=1e-6,
        )
        src_xenium = SimpleNamespace(
            slide_idx=1,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="xenium",
        )

        spatial.clear_histology_coord_caches()
        unvalidated = spatial.load_histology_coord_map_from_source(
            src_histology_unvalidated
        )
        unvalidated_stats = spatial.get_histology_coord_completion_stats(
            src_histology_unvalidated
        )
        direct = spatial.load_histology_coord_map_from_source(src_histology)
        direct_stats = spatial.get_histology_coord_completion_stats(src_histology)
        assert unvalidated == direct
        assert unvalidated_stats["alignment"] is None
        assert direct_stats["cache_hit"] is False
        assert direct == {1: (1.5, 1.5), 2: (10.5, 10.5)}
        assert direct_stats["cell_coords_id_space"] == "histology"
        assert direct_stats["csv_coordinates"] == 2
        assert direct_stats["csv_coordinate_coverage"] == 1.0
        assert direct_stats["centroid_filled"] == 0
        assert direct_stats["alignment"]["status"] == "passed"
        assert direct_stats["alignment"]["max_error_px"] <= 1e-6

        # A generic cell_id column is resolved from segmentation evidence. This
        # load must not reuse the explicitly configured histology cache entry.
        automatic = spatial.load_histology_coord_map_from_source(src_auto)
        automatic_stats = spatial.get_histology_coord_completion_stats(src_auto)
        assert automatic == direct
        assert automatic_stats["cache_hit"] is False
        assert automatic_stats["cell_coords_id_space_requested"] == "auto"
        assert automatic_stats["cell_coords_id_space"] == "histology"
        assert automatic_stats["cell_coords_id_space_resolution"] == (
            "segmentation_alignment_unique_pass"
        )
        assert automatic_stats["cell_coords_auto_candidates"]["histology"][
            "status"
        ] == "passed"
        assert automatic_stats["cell_coords_auto_candidates"]["xenium"][
            "status"
        ] == "failed"

        # Explicit legacy overrides remain isolated and retain old behavior.
        fallback = spatial.load_histology_coord_map_from_source(src_xenium)
        fallback_stats = spatial.get_histology_coord_completion_stats(src_xenium)
        assert fallback == direct
        assert fallback_stats["cell_coords_id_space"] == "xenium"
        assert fallback_stats["csv_coordinates"] == 0
        assert fallback_stats["centroid_filled"] == 2


def test_auto_detects_xenium_keyed_generic_column():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_auto_xenium_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [11, 12],
                "x_coord": [1.5, 10.5],
                "y_coord": [1.5, 10.5],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)
        src = SimpleNamespace(
            slide_idx=4,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=1e-6,
        )

        spatial.clear_histology_coord_caches()
        coordinate_map = spatial.load_histology_coord_map_from_source(src)
        stats = spatial.get_histology_coord_completion_stats(src)
        assert coordinate_map == {1: (1.5, 1.5), 2: (10.5, 10.5)}
        assert stats["cell_coords_id_space_requested"] == "auto"
        assert stats["cell_coords_id_space"] == "xenium"
        assert stats["csv_coordinates"] == 2
        assert stats["centroid_filled"] == 0
        assert stats["alignment"]["status"] == "passed"


def test_auto_prefers_explicit_histology_column():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_explicit_header_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "unused_labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [11, 12],
                "id_histology": [1, 2],
                "x_coord": [1.5, 10.5],
                "y_coord": [1.5, 10.5],
            }
        ).to_csv(fp_coords, index=False)
        fp_seg.write_bytes(b"not-opened-because-header-is-explicit")
        src = SimpleNamespace(
            slide_idx=6,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
        )

        spatial.clear_histology_coord_caches()
        coordinate_map = spatial.load_histology_coord_map_from_source(src)
        stats = spatial.get_histology_coord_completion_stats(src)
        assert coordinate_map == {1: (1.5, 1.5), 2: (10.5, 10.5)}
        assert stats["cell_coords_id_space_requested"] == "auto"
        assert stats["cell_coords_id_space"] == "histology"
        assert stats["cell_coords_id_column"] == "id_histology"
        assert stats["cell_coords_id_space_resolution"] == (
            "explicit_id_histology_column"
        )
        assert stats["alignment"] is None
        assert stats["centroid_filled"] == 0


def test_auto_ambiguous_different_maps_fails_closed():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_auto_ambiguous_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [1, 2, 11, 12],
                "x_coord": [1.5, 10.5, 1.55, 10.55],
                "y_coord": [1.5, 10.5, 1.55, 10.55],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)
        src = SimpleNamespace(
            slide_idx=5,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=0.1,
        )

        spatial.clear_histology_coord_caches()
        try:
            spatial.load_histology_coord_map_from_source(src)
        except spatial.CoordinateAlignmentError as exc:
            assert "automatic coordinate namespace is ambiguous" in str(exc)
        else:
            raise AssertionError("different dual-pass coordinate maps did not fail closed")


def test_configured_alignment_mismatch_fails_fast():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_alignment_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        # Deliberately swap the two histology-label centroids.
        pd.DataFrame(
            {
                "cell_id": [1, 2],
                "x_coord": [10.5, 1.5],
                "y_coord": [10.5, 1.5],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)
        src = SimpleNamespace(
            slide_idx=2,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="histology",
            cell_coords_validate_alignment=True,
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=0.1,
        )

        spatial.clear_histology_coord_caches()
        try:
            spatial.load_histology_coord_map_from_source(src)
        except spatial.CoordinateAlignmentError as exc:
            message = str(exc)
            assert "disagree with segmentation centroids" in message
            assert "id_space=histology" in message
        else:
            raise AssertionError("misaligned configured coordinates did not fail fast")


def test_validated_explicit_empty_join_fails_before_completion():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_empty_join_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [1, 2],
                "x_coord": [1.5, 10.5],
                "y_coord": [1.5, 10.5],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)
        src = SimpleNamespace(
            slide_idx=7,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_id_space="xenium",
            cell_coords_validate_alignment=True,
        )

        spatial.clear_histology_coord_caches()
        try:
            spatial.load_histology_coord_map_from_source(src)
        except spatial.CoordinateAlignmentError as exc:
            assert "empty map" in str(exc)
        else:
            raise AssertionError("validated empty CSV join was hidden by centroid filling")


def test_auto_validation_request_is_cache_isolated():
    with tempfile.TemporaryDirectory(prefix="ghist_coord_validation_cache_test_") as tmp:
        base = Path(tmp)
        fp_match = base / "matched_nuclei_filtered.csv"
        fp_coords = base / "cell_coords.csv"
        fp_seg = base / "labels.tif"
        pd.DataFrame(
            {"id_histology": [1, 2], "id_xenium": [11, 12]}
        ).to_csv(fp_match, index=False)
        pd.DataFrame(
            {
                "cell_id": [11, 12],
                "id_histology": [1, 2],
                "x_coord": [10.5, 1.5],
                "y_coord": [10.5, 1.5],
            }
        ).to_csv(fp_coords, index=False)
        _write_tiny_segmentation(fp_seg)
        unvalidated = SimpleNamespace(
            slide_idx=8,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
        )
        validated = SimpleNamespace(
            slide_idx=8,
            fp_nuc_sizes=str(fp_match),
            fp_nuc_seg=str(fp_seg),
            cell_coords_validate_alignment=True,
            cell_coords_alignment_sample_size=2,
            cell_coords_alignment_tolerance_px=0.1,
        )

        spatial.clear_histology_coord_caches()
        spatial.load_histology_coord_map_from_source(unvalidated)
        assert spatial.get_histology_coord_completion_stats(unvalidated)["alignment"] is None
        assert spatial._histology_coord_cache_key(unvalidated) != (
            spatial._histology_coord_cache_key(validated)
        )
        try:
            spatial.load_histology_coord_map_from_source(validated)
        except spatial.CoordinateAlignmentError as exc:
            assert "disagree with segmentation centroids" in str(exc)
        else:
            raise AssertionError("validated auto load reused an unvalidated cache entry")


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, path))
        return out
    if isinstance(value, list):
        out = {}
        for idx, child in enumerate(value):
            out.update(_flatten(child, f"{prefix}[{idx}]"))
        return out
    return {prefix: value}


def test_matched_configs_and_seed_wiring():
    on = json.loads((ROOT / "configs/breast2_ablation9/full.json").read_text())
    off = json.loads(json.dumps(on))
    off["model"]["ecrm"]["ablation_off"] = True
    flat_on, flat_off = _flatten(on), _flatten(off)
    differing = {
        key
        for key in set(flat_on) | set(flat_off)
        if flat_on.get(key) != flat_off.get(key)
    }
    assert differing == {"model.ecrm.ablation_off"}
    assert on["model"]["ecrm"]["ablation_off"] is False
    assert off["model"]["ecrm"]["ablation_off"] is True
    assert on["training"]["seed"] == off["training"]["seed"] == 20260807
    for config in (on, off):
        source = config["data_sources_train_val"][0]
        assert source["cell_coords_id_space"] == "histology"
        assert source["cell_coords_validate_alignment"] is True
    assert spatial._cell_coords_id_space(SimpleNamespace()) == "auto"

    train_source = (ROOT / "train.py").read_text()
    for required in (
        "random.seed(training_seed)",
        "np.random.seed(training_seed)",
        "torch.manual_seed(training_seed)",
        "torch.cuda.manual_seed_all(training_seed)",
        "torch.backends.cudnn.deterministic = True",
        "Reproducibility seed:",
    ):
        assert required in train_source
    seed_position = train_source.index("random.seed(training_seed)")
    device_position = train_source.index("device = utils.get_device(config.gpu_id)")
    assert device_position < train_source.index("torch.manual_seed(training_seed)")
    assert device_position < train_source.index("torch.cuda.is_available()")
    assert device_position < train_source.index("torch.cuda.manual_seed_all(training_seed)")
    assert seed_position < train_source.index("framework_cls =")
    assert seed_position < train_source.index("dataset_input.DataProcessingUnion(")


def test_real_breast2_coverage():
    data_root = os.environ.get("GHIST_DATA_ROOT")
    if not data_root:
        pytest.skip("set GHIST_DATA_ROOT to run the real Breast2 coverage test")
    base = Path(data_root) / "data_processing/data_processing_breast2"
    fp_match = base / "matched_nuclei_filtered.csv"
    fp_seg = base / "he_image_nuclei_seg.tif"
    fp_coords = base / "cell_coords.csv"
    if not (fp_match.is_file() and fp_seg.is_file() and fp_coords.is_file()):
        raise FileNotFoundError(f"real Breast2 fixture is incomplete: {base}")
    src = SimpleNamespace(
        slide_idx=3,
        fp_nuc_sizes=str(fp_match),
        fp_nuc_seg=str(fp_seg),
        cell_coords_alignment_sample_size=256,
        cell_coords_alignment_tolerance_px=1.0,
        cell_coords_alignment_window_radius_px=64,
    )
    spatial.clear_histology_coord_caches()
    matched = pd.read_csv(fp_match)
    matched_ids = set(
        pd.to_numeric(matched["id_histology"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .tolist()
    )
    coord_map = spatial.load_histology_coord_map_from_source(src)
    stats = spatial.get_histology_coord_completion_stats(src)
    covered = matched_ids.intersection(coord_map)
    assert len(matched_ids) == 86_344
    assert len(covered) == len(matched_ids)
    assert stats["cell_coords_id_space_requested"] == "auto"
    assert stats["cell_coords_id_space"] == "histology"
    assert stats["cell_coords_id_space_resolution"] == (
        "segmentation_alignment_unique_pass"
    )
    assert stats["cell_coords_id_column"] == "cell_id"
    assert stats["cell_coords_auto_candidates"]["histology"]["status"] == "passed"
    assert stats["cell_coords_auto_candidates"]["xenium"]["status"] == "failed"
    assert stats["csv_coordinates"] == 86_344
    assert stats["csv_coordinate_coverage"] == 1.0
    assert stats["centroid_filled"] == 0
    assert stats["unresolved"] == 0
    assert stats["total_coordinates"] == 86_344
    assert stats["segmentation_scan_performed"] is False
    assert stats["alignment"]["status"] == "passed"
    assert stats["alignment"]["method"] == "memory_mapped_local_windows"
    assert stats["alignment"]["sampled_labels"] == 256
    assert stats["alignment"]["max_error_px"] <= 1.0
    # A second caller must hit the completed-map cache and cannot rescan TIFF.
    again = spatial.load_histology_coord_map_from_source(src)
    assert again == coord_map
    assert spatial.get_histology_coord_completion_stats(src)["cache_hit"] is True

    wrong_namespace_src = SimpleNamespace(
        slide_idx=3,
        fp_nuc_sizes=str(fp_match),
        fp_nuc_seg=str(fp_seg),
        cell_coords_id_space="xenium",
        cell_coords_validate_alignment=True,
        cell_coords_alignment_sample_size=256,
        cell_coords_alignment_tolerance_px=1.0,
        cell_coords_alignment_window_radius_px=64,
    )
    spatial.clear_histology_coord_caches()
    try:
        spatial.load_histology_coord_map_from_source(wrong_namespace_src)
    except spatial.CoordinateAlignmentError as exc:
        message = str(exc)
        assert "id_space=xenium" in message
        assert "segmentation labels" in message
    else:
        raise AssertionError("real Breast2 Xenium-key interpretation passed alignment")
    print(
        "REAL_BREAST2_COVERAGE "
        f"{len(covered)}/{len(matched_ids)} csv={stats['csv_coordinates']} "
        f"centroid_filled={stats['centroid_filled']} unresolved={stats['unresolved']}"
    )


def run(include_real=False):
    test_csv_preserved_missing_filled_and_cached_once()
    test_completion_cache_isolated_by_missing_id_set()
    test_nonnumeric_coordinate_id_cannot_match_missing_id()
    test_histology_namespace_direct_keying_alignment_and_cache_isolation()
    test_auto_detects_xenium_keyed_generic_column()
    test_auto_prefers_explicit_histology_column()
    test_auto_ambiguous_different_maps_fails_closed()
    test_configured_alignment_mismatch_fails_fast()
    test_validated_explicit_empty_join_fails_before_completion()
    test_auto_validation_request_is_cache_isolated()
    test_matched_configs_and_seed_wiring()
    if include_real:
        test_real_breast2_coverage()
    print(f"PASS: {12 if include_real else 11} spatial/seed test groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real",
        action="store_true",
        help="validate sampled cells against the real Breast2 TIFF",
    )
    args = parser.parse_args()
    run(include_real=args.real)
