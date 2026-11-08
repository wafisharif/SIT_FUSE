"""Frame-by-frame (streaming) inference with a trained RTDBN encoder.

Reads the same YAML config as ``pretrain_rtdbn.py`` and the artifacts it writes (``<out_dir>/encoder/rtdbn.ckpt`` and
``<out_dir>/encoder/scaler.pkl``). Each frame is scaled, pushed through the stack with the recurrent state carried
over, and the top layer's hidden probabilities are saved for every frame. The output equals the layer-by-layer
sequence ``forward`` of the stack (checked by default); ``RTDBN.forward`` is the time average of the same sequence, so
the mean of the saved hidden states over a window reproduces it. No window is needed: one frame in, one state out.

Optional config block (all keys optional)::

    streaming:
      out_dir: <out_dir>/stream     # where per-episode hidden sequences are written
      verify: True                   # compare against the layer-wise sequence forward on every episode
"""

import argparse
import os

import numpy as np
import torch
import yaml

from learnergy.models.temporal.rtdbn import RTDBN
from learnergy.models.temporal.streaming import StreamingEncoder

from sit_fuse.datasets.rtdbn_data_utils import load_scaler, split_episode_files


def read_yaml(fpath):
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


def encoder_dir(yml_conf):
    return os.path.join(yml_conf["output"]["out_dir"], "encoder")


def load_stream_model(yml_conf):
    """Rebuilds the trained RTDBN and loads its scaler; returns (rtdbn, scaler)."""
    rtdbn = RTDBN(
        model=tuple(yml_conf["rtdbn"]["model_type"]),
        n_visible=int(yml_conf["rtdbn"]["n_visible"]),
        n_hidden=tuple(yml_conf["rtdbn"]["n_hidden"]),
        steps=tuple(yml_conf["rtdbn"]["gibbs_steps"]),
        learning_rate=tuple(yml_conf["encoder"]["training"]["learning_rate"]),
        momentum=tuple(yml_conf["encoder"]["training"]["momentum"]),
        decay=tuple(yml_conf["encoder"]["training"]["weight_decay"]),
        temperature=tuple(yml_conf["rtdbn"]["temp"]),
        use_gpu=False,
    )
    ckpt = os.path.join(encoder_dir(yml_conf), "rtdbn.ckpt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No RTDBN checkpoint at {ckpt}. Run pretrain_rtdbn.py first.")
    rtdbn.load_state_dict(torch.load(ckpt, map_location="cpu"))
    rtdbn.eval()
    scaler = load_scaler(os.path.join(encoder_dir(yml_conf), "scaler.pkl"))
    return rtdbn, scaler


def load_frames(path, scaler):
    """Loads one raw episode (frames, n_visible) and applies the training scaler."""
    raw = np.load(path).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"{path}: expected an array of shape (frames, n_visible), got {raw.shape}.")
    return scaler.transform(raw).astype(np.float32)


def stream_episode(rtdbn, frames, verify=True):
    """Streams scaled frames one at a time; returns (hidden[frames, n_hidden_top], max_abs_diff_vs_sequence_forward)."""
    enc = StreamingEncoder(rtdbn)
    hidden = enc.run(torch.as_tensor(frames).unsqueeze(0))[0].numpy()
    diff = None
    if verify:
        try:
            diff = enc.max_abs_difference(torch.as_tensor(frames).unsqueeze(0))
        except TypeError:
            # Some layer classes have no batch forward that takes the recurrent state; nothing to compare with.
            diff = None
    return hidden, diff


def default_episode_files(yml_conf):
    """Episodes to stream by default: the test split, or the validation split when there is no test split."""
    data = yml_conf["data"]
    _, val_files, test_files = split_episode_files(
        data["data_dir"],
        float(data["val_percent"]),
        float(data.get("test_percent", 0.0)),
        seed=int(data.get("split_seed", 42)),
    )
    return test_files if len(test_files) > 0 else val_files


def stream_encode(yml_conf, episode_files=None):
    """Streams every episode and writes ``<name>_hidden.npy``; returns {file: max_abs_diff_vs_forward}."""
    cfg = yml_conf.get("streaming", {}) or {}
    out_dir = cfg.get("out_dir", os.path.join(yml_conf["output"]["out_dir"], "stream"))
    verify = bool(cfg.get("verify", True))
    os.makedirs(out_dir, exist_ok=True)
    rtdbn, scaler = load_stream_model(yml_conf)
    files = episode_files if episode_files else default_episode_files(yml_conf)
    summary = {}
    for path in files:
        hidden, diff = stream_episode(rtdbn, load_frames(path, scaler), verify=verify)
        name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(out_dir, f"{name}_hidden.npy"), hidden)
        summary[path] = diff
        print(f"{name}: {hidden.shape[0]} frames -> {hidden.shape[1]} hidden units, max |stream - forward| = {diff}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file used for RTDBN training.")
    parser.add_argument("-e", "--episodes", nargs="*", default=None, help="Episode .npy files (default: test split).")
    args = parser.parse_args()
    stream_encode(read_yaml(args.yaml), args.episodes)
