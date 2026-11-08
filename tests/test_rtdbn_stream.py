"""Streaming encode and novelty monitor regressions; tiny random RTDBN, synthetic episodes, CPU only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from joblib import dump
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learnergy.models.temporal.rtdbn import RTDBN  # noqa: E402

from sit_fuse.inference import rtdbn_novelty, rtdbn_stream  # noqa: E402

N_VISIBLE = 6


def make_config(root, n_hidden=(5,), model_type=None):
    n_layers = len(n_hidden)
    model_type = model_type or ["variance_gaussian"] * n_layers
    return {
        "data": {"data_dir": os.path.join(root, "data"), "seq_len": 20, "val_percent": 0.25, "test_percent": 0.25,
                 "split_seed": 1, "fill_value": -999999.0},
        "output": {"out_dir": os.path.join(root, "out")},
        "rtdbn": {"model_type": model_type, "n_visible": N_VISIBLE, "n_hidden": list(n_hidden),
                  "gibbs_steps": [1] * n_layers, "temp": [1.0] * n_layers},
        "encoder": {"training": {"learning_rate": [0.001] * n_layers, "momentum": [0.0] * n_layers,
                                 "weight_decay": [0.0] * n_layers}},
    }


def write_fixture(root, n_hidden=(5,), seed=0):
    """Writes episodes, a fitted scaler and a randomly initialised (but non-trivial) RTDBN checkpoint."""
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, "data"))
    for i in range(8):
        np.save(os.path.join(root, "data", f"episode_{i:03d}.npy"), rng.randn(150, N_VISIBLE).astype(np.float32))
    conf = make_config(root, n_hidden)
    scaler = StandardScaler().fit(rng.randn(500, N_VISIBLE))
    enc_dir = os.path.join(root, "out", "encoder")
    os.makedirs(enc_dir)
    dump(scaler, os.path.join(enc_dir, "scaler.pkl"))
    torch.manual_seed(seed)
    rtdbn = RTDBN(model=tuple(conf["rtdbn"]["model_type"]), n_visible=N_VISIBLE, n_hidden=tuple(n_hidden),
                  steps=(1,) * len(n_hidden), learning_rate=(0.001,) * len(n_hidden), momentum=(0.0,) * len(n_hidden),
                  decay=(0.0,) * len(n_hidden), temperature=(1.0,) * len(n_hidden))
    with torch.no_grad():
        for layer in rtdbn.models:
            layer.W_prime.normal_(0.0, 0.5)
            layer.h0.normal_(0.0, 0.5)
    torch.save(rtdbn.state_dict(), os.path.join(enc_dir, "rtdbn.ckpt"))
    return conf, rtdbn, scaler


class StreamEncodeTest(unittest.TestCase):
    def test_stream_equals_sequence_forward_one_layer(self):
        with tempfile.TemporaryDirectory() as root:
            conf, rtdbn, scaler = write_fixture(root)
            files = sorted(Path(root, "data").glob("episode_*.npy"))[:2]
            summary = rtdbn_stream.stream_encode(conf, [str(f) for f in files])
            for f in files:
                saved = np.load(Path(root, "out", "stream", f"{f.stem}_hidden.npy"))
                frames = torch.as_tensor(scaler.transform(np.load(f)).astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    ref = rtdbn.models[0].forward(frames)[0].numpy()
                    pooled = rtdbn.forward(frames[:, :30])[0].numpy()
                self.assertTrue(np.allclose(saved, ref, atol=1e-6))
                self.assertTrue(np.allclose(saved[:30].mean(axis=0), pooled, atol=1e-6))  # RTDBN.forward = time mean
                self.assertLess(summary[str(f)], 1e-6)

    def test_stream_two_layers_and_default_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            conf, rtdbn, scaler = write_fixture(root, n_hidden=(5, 4))
            summary = rtdbn_stream.stream_encode(conf)
            self.assertEqual(len(summary), 2)  # test split: 25% of 8 files
            self.assertTrue(all(v < 1e-6 for v in summary.values()))
            hidden = np.load(next(Path(root, "out", "stream").glob("*_hidden.npy")))
            self.assertEqual(hidden.shape, (150, 4))

    def test_missing_checkpoint_and_bad_episode_shape(self):
        with tempfile.TemporaryDirectory() as root:
            conf, _, scaler = write_fixture(root)
            np.save(os.path.join(root, "bad.npy"), np.zeros((4, 3, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                rtdbn_stream.load_frames(os.path.join(root, "bad.npy"), scaler)
            os.remove(os.path.join(root, "out", "encoder", "rtdbn.ckpt"))
            with self.assertRaises(FileNotFoundError):
                rtdbn_stream.load_stream_model(conf)


class NoveltyTest(unittest.TestCase):
    def test_calibrate_then_score(self):
        with tempfile.TemporaryDirectory() as root:
            conf, rtdbn, scaler = write_fixture(root)
            familiar = [str(f) for f in sorted(Path(root, "data").glob("episode_*.npy"))[:6]]
            record = rtdbn_novelty.calibrate(conf, familiar)
            self.assertEqual(record["n_windows"], 6 * 5)
            saved = json.load(open(rtdbn_novelty.novelty_settings(conf)["threshold_path"]))
            self.assertAlmostEqual(saved["threshold"], record["threshold"])

            shifted = os.path.join(root, "shifted.npy")
            np.save(shifted, (np.load(familiar[0]) * 4.0 + 6.0).astype(np.float32))
            out = rtdbn_novelty.score_episodes(conf, [familiar[0], shifted])
            self.assertGreater(out[shifted], 0.9)
            self.assertLess(out[familiar[0]], 0.5)
            res = np.load(Path(root, "out", "novelty", "shifted_novelty.npz"))
            self.assertEqual(res["score"].shape, (150,))
            self.assertFalse(res["alarm"][:29].any())  # no alarm before a full window

    def test_default_calibration_uses_validation_split_and_requires_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            conf, _, _ = write_fixture(root)
            with self.assertRaises(FileNotFoundError):
                rtdbn_novelty.score_episodes(conf, [os.path.join(root, "data", "episode_000.npy")])
            record = rtdbn_novelty.calibrate(conf)
            self.assertEqual(len(record["files"]), 2)  # 25% of 8 files

    def test_score_frames_matches_library_window_scores(self):
        with tempfile.TemporaryDirectory() as root:
            conf, rtdbn, scaler = write_fixture(root)
            frames = rtdbn_stream.load_frames(os.path.join(root, "data", "episode_000.npy"), scaler)
            scores, _ = rtdbn_novelty.score_frames(rtdbn, frames, 30, None)
            from learnergy.models.temporal.novelty import window_free_energy
            first = window_free_energy(rtdbn, frames[None, :30])[0]
            self.assertAlmostEqual(float(scores[29]), float(first), places=4)


if __name__ == "__main__":
    unittest.main()
