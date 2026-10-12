import json
from pathlib import Path

from brats_debate.config import file_hash, fingerprint, load_config, protocol


REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / "docs" / "beta1" / "manifest.json"
CONFIG_PATH = REPO / "configs" / "brats_controller.yaml"
SPLIT_PATH = REPO / "configs" / "splits" / "brats2023_gli_cases.json"

EXPECTED_HASHES = {
    "cnn": "e8fbde5084353dc5aa499626d208fad7c7c4595e23376daa7d98a4890259697c",
    "transformer": "dd77255376fb0831e7b91795dff3dbf1bba647b3e305df9ef5dfa21d6d55b64b",
    "boundary": "dd360d8d6430acd4eff116b181ed3b83b2144fffd16c0bf8c0baaa913f00b79c",
    "highres": "15bd9b270ff3ca160e7a4f07a93809e1e5ad9c7e25b352966df5940d639b83c0",
    "controller": "86b604082a934a1d4f4a68467a88d16d268ae7653c552dcfc6c3087ba1d79fe2",
}


def test_frozen_beta1_manifest_protocol_and_splits():
    manifest = json.loads(MANIFEST_PATH.read_text())
    config = load_config(CONFIG_PATH)
    splits = json.loads(SPLIT_PATH.read_text())

    assert manifest["status"] == "beta1_complete"
    assert manifest["branch"] == "main"
    assert manifest["selected_controller_epoch"] == 19
    assert manifest["splits"] == {
        "expert_train": 682,
        "controller_train": 248,
        "validation": 191,
        "test": 130,
    }
    assert {name: value["sha256"] for name, value in manifest["checkpoints"].items()} == EXPECTED_HASHES
    assert file_hash(CONFIG_PATH) == manifest["config_sha256"]
    assert file_hash(SPLIT_PATH) == manifest["split_file_sha256"]
    assert fingerprint(protocol(config)) == manifest["protocol_hash"]
    assert fingerprint(splits) == manifest["split_hash"]


def test_present_beta1_checkpoints_match_manifest_without_requiring_binaries():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for name, record in manifest["checkpoints"].items():
        checkpoint = REPO / record["path"]
        if checkpoint.is_file():
            assert file_hash(checkpoint) == EXPECTED_HASHES[name]


def test_frozen_beta1_reported_results_are_not_reinterpreted():
    manifest = json.loads(MANIFEST_PATH.read_text())
    results = manifest["final_test"]["brats_mean_dice"]
    assert results["cnn"] == 0.8684446209600831
    assert results["transformer"] == 0.8660222073649182
    assert results["boundary"] == 0.8634439675434475
    assert results["highres"] == 0.81344098171662
    assert results["final"] == 0.8734352777446935
    assert results["vote"] == 0.8731181744723431
