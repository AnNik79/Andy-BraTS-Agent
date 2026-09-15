import copy
import numpy as np
import pytest
import torch
from brats_debate.config import seed_everything
from brats_debate.experts import build_expert
from brats_debate.training.train_expert import training_output, expert_loss
from brats_debate.data.brats_dataset import discover_patients, BraTSPatches, make_patch_loader, load_patient, convert_labels
from brats_debate.inference.patches import sliding_probabilities
from brats_debate.data.preprocessing import sample_center


def test_minimal_training_forward_step_and_rng_match(cfg):
    seed_everything(77)
    original = build_expert('cnn', cfg).train()
    minimal = copy.deepcopy(original)
    x = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    results = []
    for model, optimized in [(original, False), (minimal, True)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
        seed_everything(88)
        out = training_output(model, x, 'cnn', dict(cfg, cnn_probability_only_training=optimized))
        loss = expert_loss(out, target)
        loss.backward()
        optimizer.step()
        results.append((out['probabilities'].detach(), torch.get_rng_state()))
    for a, b in zip(results[0], results[1]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(original.parameters(), minimal.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('batch', [1,2,3,8])
@pytest.mark.parametrize('shape', [(11,9,7),(5,6,7)])
def test_batched_validation_preserves_probabilities(cfg, batch, shape):
    model = build_expert('cnn', cfg).eval()
    image = torch.randn(1, 4, *shape)
    reference = sliding_probabilities(model, image, (8,8,8), 'cpu')
    actual = sliding_probabilities(model, image, (8,8,8), 'cpu', batch_size=batch,
                                   accumulation_device='model', cached_geometry=True)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-5)
    assert actual.grad_fn is None
    torch.testing.assert_close(actual.argmax(1), reference.argmax(1))


def test_thread_prefetch_keeps_samples_and_rng_exact(cfg):
    records = discover_patients(cfg)
    dataset = BraTSPatches(records, cfg)
    indices = [2,0,3,1,2]
    results = []
    for enabled in [False, True]:
        seed_everything(55)
        batches = list(make_patch_loader(dataset, indices, dict(cfg, prefetch_patients=enabled), 123))
        results.append((batches, torch.get_rng_state(), np.random.get_state()))
    for a, b in zip(results[0][0], results[1][0]):
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(results[0][1],results[1][1],atol=0,rtol=0)
    assert np.array_equal(results[0][2][1], results[1][2][1])
    assert results[0][2][2:] == results[1][2][2:]


@pytest.mark.parametrize('specialist',[False,True])
def test_flat_sampler_exact_centers_and_rng(specialist):
    labels=np.zeros((13,12,11),dtype=np.int64)
    labels[4:7,2:5,6:9]=2
    for seed in range(20):
        a,b=np.random.default_rng(seed),np.random.default_rng(seed)
        assert sample_center(labels,a,specialist)==sample_center(labels,b,specialist,flat=True)
        assert a.bit_generator.state==b.bit_generator.state


def test_lossless_patient_cache_values_geometry_and_corruption(cfg,tmp_path):
    pytest.importorskip('lz4.frame')
    record=discover_patients(cfg)[0]
    expected=load_patient(record,cfg)
    cached=dict(cfg,patient_cache_dir=str(tmp_path/'cache'),patient_cache_max_gib=.02)
    load_patient(record,cached)
    actual=load_patient(record,cached)
    for key in ('image','label','affine'):
        assert np.array_equal(expected[key],actual[key])
    assert expected['header'].binaryblock==actual['header'].binaryblock
    assert expected['spacing']==actual['spacing']
    from brats_debate.data.patient_cache import cache_path
    cache_path(record,cached).write_bytes(b'broken')
    repaired=load_patient(record,cached)
    assert np.array_equal(expected['image'],repaired['image'])


def test_fast_mapping_retains_unknown_label_checks():
    labels=np.array([[[0.,1.,2.,3.]]],dtype=np.float32)
    mapping={0:0,1:2,2:3,3:1}
    assert np.array_equal(convert_labels(labels,mapping),convert_labels(labels,mapping,fast=True))
    for invalid in [4.,.5,float('nan')]:
        with pytest.raises(ValueError):
            convert_labels(np.array([[[invalid]]]),mapping,fast=True)


def test_numpy_argmax_preserves_tie_breaking():
    probabilities=torch.tensor([[[[[.5,.1,.0]]]],[[[[.5,.3,.0]]]],[[[[.0,.3,1.]]]],[[[[.0,.3,.0]]]]]).reshape(1,4,1,1,3)
    assert np.array_equal(probabilities.argmax(1)[0].numpy(), probabilities.numpy()[0].argmax(axis=0))
