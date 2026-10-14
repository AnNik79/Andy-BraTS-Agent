import json
from pathlib import Path
import sys

import pytest

from brats_debate.config import fingerprint


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import beta2_fit_collection as collection  # noqa: E402


def test_frozen_partition_manifest_is_sealed_and_grouped():
    path = ROOT / "configs" / "beta2" / "subject_partitions.json"
    manifest = json.loads(path.read_text())
    payload = dict(manifest)
    recorded_hash = payload.pop("manifest_hash")
    assert recorded_hash == fingerprint(payload)
    assert manifest["case_counts"] == {
        "fit": 111,
        "calibration": 41,
        "evaluation": 39,
    }
    assert manifest["subject_counts"] == {
        "fit": 101,
        "calibration": 35,
        "evaluation": 34,
    }
    all_cases = [
        case_id
        for role in ("fit", "calibration", "evaluation")
        for case_id in manifest["partitions"][role]
    ]
    assert len(all_cases) == len(set(all_cases)) == 191
    subject_owner = {}
    for role, subjects in manifest["subject_partitions"].items():
        for subject in subjects:
            assert subject not in subject_owner
            subject_owner[subject] = role


def test_deterministic_fit_selection_excludes_processed_and_repeated_subjects():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
    }
    assert (
        collection.select_fit_cases(manifest, case_to_subject, processed)
        == collection.EXPECTED_CASES
    )
    subjects = [case_to_subject[case_id] for case_id in collection.EXPECTED_CASES]
    assert len(subjects) == len(set(subjects))


def test_second_batch_is_next_ten_unprocessed_fit_cases():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        "BraTS-GLI-00049-000",
        "BraTS-GLI-00056-000",
        "BraTS-GLI-00101-000",
    }
    profile = collection.BATCH_PROFILES["fit_collection_batch_20261013_02"]
    selected = collection.select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=10,
        distinct_subjects=False,
    )
    assert selected == profile["cases"]
    assert all(case_id in manifest["partitions"]["fit"] for case_id in selected)


def test_third_batch_is_next_ten_unprocessed_fit_cases():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        "BraTS-GLI-00049-000",
        "BraTS-GLI-00056-000",
        "BraTS-GLI-00056-001",
        "BraTS-GLI-00101-000",
        "BraTS-GLI-00104-000",
        "BraTS-GLI-00112-000",
        "BraTS-GLI-00133-000",
        "BraTS-GLI-00167-000",
        "BraTS-GLI-00188-000",
        "BraTS-GLI-00222-000",
        "BraTS-GLI-00230-000",
        "BraTS-GLI-00235-000",
        "BraTS-GLI-00240-000",
    }
    profile = collection.BATCH_PROFILES["fit_collection_batch_20261013_03"]
    selected = collection.select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=10,
        distinct_subjects=False,
    )
    assert selected == profile["cases"]
    assert all(case_id in manifest["partitions"]["fit"] for case_id in selected)
    assert len({case_to_subject[case_id] for case_id in selected}) == 10


def test_fourth_batch_is_next_ten_unprocessed_fit_cases():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        *collection.BATCH_PROFILES[collection.DEFAULT_BATCH_ID]["cases"],
        *collection.BATCH_PROFILES["fit_collection_batch_20261013_02"]["cases"],
        *collection.BATCH_PROFILES["fit_collection_batch_20261013_03"]["cases"],
    }
    profile = collection.BATCH_PROFILES["fit_collection_batch_20261013_04"]
    selected = collection.select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=10,
        distinct_subjects=False,
    )
    assert selected == profile["cases"]
    assert all(case_id in manifest["partitions"]["fit"] for case_id in selected)
    assert {
        case_to_subject[case_id] for case_id in selected
    }.isdisjoint(manifest["subject_partitions"]["calibration"])
    assert {
        case_to_subject[case_id] for case_id in selected
    }.isdisjoint(manifest["subject_partitions"]["evaluation"])


def test_post_restart_batch_is_next_five_unprocessed_fit_cases():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        *collection.BATCH_PROFILES[collection.DEFAULT_BATCH_ID]["cases"],
        *collection.BATCH_PROFILES["fit_collection_batch_20261013_02"]["cases"],
        *collection.BATCH_PROFILES["fit_collection_batch_20261013_03"]["cases"],
        *collection.BATCH_PROFILES["fit_collection_batch_20261013_04"]["cases"],
    }
    profile = collection.BATCH_PROFILES["fit_collection_batch_20261014_01"]
    selected = collection.select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=5,
        distinct_subjects=False,
    )
    assert selected == profile["cases"]
    assert all(case_id in manifest["partitions"]["fit"] for case_id in selected)
    selected_subjects = {case_to_subject[case_id] for case_id in selected}
    assert selected_subjects.isdisjoint(
        manifest["subject_partitions"]["calibration"]
    )
    assert selected_subjects.isdisjoint(manifest["subject_partitions"]["evaluation"])


def test_second_five_case_batch_is_next_unprocessed_fit_cases():
    manifest = json.loads(
        (ROOT / "configs" / "beta2" / "subject_partitions.json").read_text()
    )
    case_to_subject = {
        case_id: case_id.rsplit("-", 1)[0]
        for case_id in manifest["allowed_case_ids"]
    }
    prior_profiles = (
        collection.DEFAULT_BATCH_ID,
        "fit_collection_batch_20261013_02",
        "fit_collection_batch_20261013_03",
        "fit_collection_batch_20261013_04",
        "fit_collection_batch_20261014_01",
    )
    processed = {
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00019-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00031-000",
        *(
            case_id
            for profile_id in prior_profiles
            for case_id in collection.BATCH_PROFILES[profile_id]["cases"]
        ),
    }
    profile = collection.BATCH_PROFILES["fit_collection_batch_20261014_02"]
    selected = collection.select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=5,
        distinct_subjects=False,
    )
    assert selected == profile["cases"]
    assert all(case_id in manifest["partitions"]["fit"] for case_id in selected)
    selected_subjects = {case_to_subject[case_id] for case_id in selected}
    assert selected_subjects.isdisjoint(
        manifest["subject_partitions"]["calibration"]
    )
    assert selected_subjects.isdisjoint(manifest["subject_partitions"]["evaluation"])


def test_collected_case_discovery_requires_complete_validation_provenance(tmp_path):
    valid = {
        "complete": True,
        "patient_id": "validation-case",
        "provenance": {
            "format": "beta2_error_prediction_case_v1",
            "source_split": "validation",
        },
    }
    invalid = {
        "complete": True,
        "patient_id": "test-case",
        "provenance": {
            "format": "beta2_error_prediction_case_v1",
            "source_split": "test",
        },
    }
    (tmp_path / "valid.summary.json").write_text(json.dumps(valid))
    (tmp_path / "invalid.summary.json").write_text(json.dumps(invalid))
    assert collection.discover_collected_cases(tmp_path) == {"validation-case"}


def test_resume_preserves_original_scientific_provenance(tmp_path):
    selected = ("case-a", "case-b")
    original = {
        "format": "beta2_stage_b_fit_collection_v1",
        "selected_cases": list(selected),
        "partition_role": "fit",
        "cases": [],
        "total_seconds": 123.0,
        "model_load_seconds": 0.5,
        "provenance": {
            "protocol_hash": "protocol",
            "split_hash": "split",
            "partition_role": "fit",
            "code_commit": "original-runner",
        },
    }
    (tmp_path / "collection.summary.json").write_text(json.dumps(original))
    candidate = {
        **original["provenance"],
        "code_commit": "resume-runner",
    }
    state = collection.load_resume_state(tmp_path, selected, candidate)
    assert state["completed"] == []
    assert state["prior_elapsed_seconds"] == 123.0
    assert state["provenance"]["code_commit"] == "original-runner"

    candidate["protocol_hash"] = "changed"
    with pytest.raises(ValueError, match="scientific provenance"):
        collection.load_resume_state(tmp_path, selected, candidate)
