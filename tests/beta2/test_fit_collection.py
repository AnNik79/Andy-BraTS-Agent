import json
from pathlib import Path
import sys

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
