from pathlib import Path
import torch
import pytest
from brats_debate.data.brats_dataset import discover_patients, make_splits
from brats_debate.training.train_expert import train_expert
from brats_debate.training.checkpoint import load_training_checkpoint, cpu_snapshot
from brats_debate.experts import build_expert
from brats_debate.inference.patches import sliding_probabilities, sliding_predict


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.device.type == "cpu" and not a.requires_grad and a.grad_fn is None
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("checkpoint_name", ["progress.pt", "epoch_train_complete.pt", "last.pt"])
@pytest.mark.parametrize("optimized", [False, True])
def test_resume_matches_uninterrupted_weights_optimizer_rng(cfg, tmp_path, checkpoint_name, optimized):
    cfg.update(epochs=2, patches_per_patient=4, save_every_steps=2, gradient_accumulation=2,
               num_workers=0, torch_threads=2, validation_max_cases=None, max_train_steps=None)
    cfg.update(cnn_probability_only_training=optimized, prefetch_patients=optimized,
               flat_candidate_sampling=optimized, fast_label_mapping=optimized, numpy_validation_argmax=optimized,
               validation_performance=dict(batch_size=2, accumulation_device="cpu", cached_geometry=True) if optimized else {})
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    cfg["output_dir"] = str(tmp_path / "reference")
    train_expert("cnn", records, splits, cfg)
    reference = torch.load(Path(cfg["output_dir"]) / "checkpoints/cnn/last.pt", weights_only=True)
    cfg.update(output_dir=str(tmp_path / "first_epoch"), epochs=1)
    train_expert("cnn", records, splits, cfg)
    source = Path(cfg["output_dir"]) / "checkpoints/cnn" / checkpoint_name
    cfg.update(output_dir=str(tmp_path / "resumed"), epochs=2)
    train_expert("cnn", records, splits, cfg, resume=source)
    resumed = torch.load(Path(cfg["output_dir"]) / "checkpoints/cnn/last.pt", weights_only=True)
    assert resumed["completed_epoch"] == reference["completed_epoch"] == 2
    assert resumed["global_step"] == reference["global_step"] == 4
    for key in ("state_dict", "optimizer_state_dict", "rng_state", "best_validation_metric"):
        assert_tree_equal(reference[key], resumed[key])
    assert isinstance(resumed["running_loss"], float)
    assert resumed["scheduler_state_dict"] is None


def test_legacy_checkpoint_is_not_false_resume(cfg, tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, path)
    model = build_expert("cnn", cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(ValueError, match="Legacy weights-only"):
        load_training_checkpoint(path, model, optimizer, scaler, cfg, {}, "cnn", torch.device("cpu"))


def test_probability_only_validation_matches_standard_output(cfg):
    model = build_expert("cnn", cfg).eval()
    image = torch.randn(1, 4, 11, 9, 7)
    expected = sliding_predict(model, image, (8, 8, 8), "cpu")["probabilities"]
    actual = sliding_probabilities(model, image, (8, 8, 8), "cpu")
    torch.testing.assert_close(expected, actual)
    assert actual.device.type == "cpu" and actual.grad_fn is None


def test_snapshot_has_no_aliases_or_graph():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    snapshot = cpu_snapshot({"value": parameter})
    assert not snapshot["value"].requires_grad
    assert snapshot["value"].data_ptr() != parameter.data_ptr()


def test_resume_into_new_directory_preserves_historical_best(cfg, tmp_path):
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    cfg.update(output_dir=str(tmp_path / 'original'), epochs=1)
    original_best = train_expert('cnn', records, splits, cfg)
    source = original_best.parent / 'last.pt'
    cfg['output_dir'] = str(tmp_path / 'relocated')
    resumed_best = train_expert('cnn', records, splits, cfg, resume=source)
    assert original_best.read_bytes() == resumed_best.read_bytes()
