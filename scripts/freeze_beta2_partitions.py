"""Create or verify the immutable Stage B subject-grouped partition manifest."""

import argparse
import json
from pathlib import Path

import yaml

from brats_debate.config import fingerprint, load_config, protocol
from brats_debate.data.brats_dataset import (
    discover_patients,
    make_splits,
    subject_id,
)
from brats_debate.training.train_controller import atomic_write_text
from brats_debate.uncertainty.partitions import (
    make_subject_partitions,
    validate_subject_partitions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "configs" / "beta2" / "subject_partitions.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "brats_controller.yaml"
    )
    parser.add_argument(
        "--stage-b-config",
        type=Path,
        default=ROOT / "configs" / "beta2" / "foundation.yaml",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def build_manifest(cfg, stage_b_cfg, records, splits):
    case_to_subject = {
        record["patient_id"]: subject_id(record, cfg) for record in records
    }
    test_subjects = sorted(
        {case_to_subject[case_id] for case_id in splits["test"]}
    )
    manifest = make_subject_partitions(
        case_to_subject,
        splits["validation"],
        forbidden_case_ids=splits["test"],
        forbidden_subject_ids=test_subjects,
        seed=int(stage_b_cfg["partition_seed"]),
        fractions=stage_b_cfg["partition_fractions"],
    )
    manifest.update(
        protocol_hash=fingerprint(protocol(cfg)),
        split_hash=fingerprint(splits),
        ordering="lexicographic_case_id_within_role",
        grouping="patient_group_regex_subject",
        test_policy="identities_only_for_exclusion; no test images or labels",
    )
    manifest["manifest_hash"] = fingerprint(manifest)
    validate_frozen_manifest(
        manifest,
        cfg=cfg,
        records=records,
        splits=splits,
        stage_b_cfg=stage_b_cfg,
    )
    return manifest


def validate_frozen_manifest(manifest, *, cfg, records, splits, stage_b_cfg):
    case_to_subject = {
        record["patient_id"]: subject_id(record, cfg) for record in records
    }
    test_subjects = sorted(
        {case_to_subject[case_id] for case_id in splits["test"]}
    )
    validate_subject_partitions(
        manifest,
        case_to_subject,
        forbidden_case_ids=splits["test"],
        forbidden_subject_ids=test_subjects,
    )
    if manifest.get("protocol_hash") != fingerprint(protocol(cfg)):
        raise ValueError("Stage B partition manifest protocol hash is stale")
    if manifest.get("split_hash") != fingerprint(splits):
        raise ValueError("Stage B partition manifest split hash is stale")
    if manifest.get("seed") != int(stage_b_cfg["partition_seed"]):
        raise ValueError("Stage B partition seed differs from configuration")
    if manifest.get("fractions") != stage_b_cfg["partition_fractions"]:
        raise ValueError("Stage B partition fractions differ from configuration")
    payload = dict(manifest)
    recorded_hash = payload.pop("manifest_hash", None)
    if recorded_hash != fingerprint(payload):
        raise ValueError("Stage B partition manifest fingerprint is invalid")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    stage_b_cfg = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    expected = build_manifest(cfg, stage_b_cfg, records, splits)
    if args.check:
        actual = json.loads(args.output.read_text())
        validate_frozen_manifest(
            actual,
            cfg=cfg,
            records=records,
            splits=splits,
            stage_b_cfg=stage_b_cfg,
        )
        if actual != expected:
            raise ValueError("frozen Stage B partition manifest is not deterministic")
        print(
            json.dumps(
                {
                    "valid": True,
                    "path": str(args.output),
                    "manifest_hash": actual["manifest_hash"],
                    "case_counts": actual["case_counts"],
                    "subject_counts": actual["subject_counts"],
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        actual = json.loads(args.output.read_text())
        if actual != expected:
            raise RuntimeError("refusing to replace a different partition manifest")
        return
    atomic_write_text(args.output, json.dumps(expected, indent=2, allow_nan=False))
    print(args.output)


if __name__ == "__main__":
    main()
