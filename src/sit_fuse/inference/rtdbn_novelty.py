"""Free-energy novelty monitor for a trained RTDBN encoder: calibrate a threshold, then score streams live.

Uses the first layer's free energy (the RTRBM energy of each frame given the carried hidden state), averaged over the
last ``window`` frames. See RESULTS_14 in SIT-FUSE-RTRBM/notes: a threshold taken from other subjects did not transfer
(nominal 5% false alarms became 10-25%), so calibrate on familiar data from the person or system being monitored.

Optional config block (all keys optional)::

    streaming:
      novelty:
        quantile: 0.95                 # alarm level = this quantile of familiar-window scores
        window: 30                     # frames averaged into the score
        threshold_path: <out_dir>/encoder/novelty_threshold.json
        out_dir: <out_dir>/novelty     # where per-episode scores are written
"""

import argparse
import json
import os

import numpy as np

from learnergy.models.temporal.novelty import NoveltyMonitor, StreamingNoveltyScorer

from sit_fuse.datasets.rtdbn_data_utils import split_episode_files
from sit_fuse.inference.rtdbn_stream import encoder_dir, load_frames, load_stream_model, read_yaml


def novelty_settings(yml_conf):
    cfg = (yml_conf.get("streaming", {}) or {}).get("novelty", {}) or {}
    out = {
        "quantile": float(cfg.get("quantile", 0.95)),
        "window": int(cfg.get("window", 30)),
        "threshold_path": cfg.get("threshold_path", os.path.join(encoder_dir(yml_conf), "novelty_threshold.json")),
        "out_dir": cfg.get("out_dir", os.path.join(yml_conf["output"]["out_dir"], "novelty")),
    }
    return out


def calibration_windows(files, scaler, window):
    """Non-overlapping blocks of `window` consecutive frames from each familiar episode."""
    blocks = []
    for path in files:
        frames = load_frames(path, scaler)
        for k in range(len(frames) // window):
            blocks.append(frames[k * window:(k + 1) * window])
    if len(blocks) == 0:
        raise ValueError(f"No calibration episode has {window} or more frames.")
    return np.stack(blocks)


def calibrate(yml_conf, familiar_files=None):
    """Sets and saves the alarm threshold from familiar, non-training episodes.

    Without `familiar_files` the validation split is used (held out from training), which is the fallback; calibrate on
    the monitored subject's own familiar episodes when possible.
    """
    cfg = novelty_settings(yml_conf)
    rtdbn, scaler = load_stream_model(yml_conf)
    if not familiar_files:
        data = yml_conf["data"]
        _, familiar_files, _ = split_episode_files(
            data["data_dir"],
            float(data["val_percent"]),
            float(data.get("test_percent", 0.0)),
            seed=int(data.get("split_seed", 42)),
        )
    windows = calibration_windows(familiar_files, scaler, cfg["window"])
    monitor = NoveltyMonitor(rtdbn, quantile=cfg["quantile"])
    threshold = monitor.calibrate(windows)
    record = {
        "threshold": threshold,
        "quantile": cfg["quantile"],
        "window": cfg["window"],
        "n_windows": int(len(windows)),
        "files": [os.path.basename(f) for f in familiar_files],
    }
    os.makedirs(os.path.dirname(cfg["threshold_path"]), exist_ok=True)
    with open(cfg["threshold_path"], "w") as f:
        json.dump(record, f, indent=2)
    print(f"Threshold {threshold:.4f} from {len(windows)} windows -> {cfg['threshold_path']}")
    return record


def score_frames(rtdbn, frames, window, threshold):
    """Streams scaled frames one at a time; returns (moving-average score[frames], alarm[frames])."""
    scorer = StreamingNoveltyScorer(rtdbn, window=window, threshold=threshold)
    scorer.reset(1)
    scores = np.zeros(len(frames), dtype=np.float32)
    alarms = np.zeros(len(frames), dtype=bool)
    for t, frame in enumerate(frames):
        score, alarm = scorer.step(frame)
        scores[t], alarms[t] = score[0], alarm[0]
    return scores, alarms


def score_episodes(yml_conf, episode_files):
    """Scores episodes with the saved threshold; writes ``<name>_novelty.npz`` (score, alarm) per episode."""
    cfg = novelty_settings(yml_conf)
    if not os.path.exists(cfg["threshold_path"]):
        raise FileNotFoundError(f"No threshold at {cfg['threshold_path']}. Run the calibrate command first.")
    with open(cfg["threshold_path"]) as f:
        record = json.load(f)
    rtdbn, scaler = load_stream_model(yml_conf)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    summary = {}
    for path in episode_files:
        scores, alarms = score_frames(rtdbn, load_frames(path, scaler), record["window"], record["threshold"])
        name = os.path.splitext(os.path.basename(path))[0]
        np.savez(os.path.join(cfg["out_dir"], f"{name}_novelty.npz"), score=scores, alarm=alarms)
        scored = alarms[record["window"] - 1:]
        summary[path] = float(scored.mean()) if len(scored) else float("nan")
        print(f"{name}: alarm on {summary[path]:.1%} of frames after the first {record['window']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["calibrate", "score"])
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file used for RTDBN training.")
    parser.add_argument("-e", "--episodes", nargs="*", default=None,
                        help="calibrate: familiar episodes (default: validation split); score: episodes to score.")
    args = parser.parse_args()
    conf = read_yaml(args.yaml)
    if args.command == "calibrate":
        calibrate(conf, args.episodes)
    else:
        if not args.episodes:
            parser.error("score needs -e with at least one episode file.")
        score_episodes(conf, args.episodes)
