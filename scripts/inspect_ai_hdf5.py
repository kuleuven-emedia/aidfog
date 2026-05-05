"""Inspect AI HDF5 output to diagnose why no cues fired.

Walks the file structure, then prints fog-probability statistics
computed from softmax over the stored logits.
"""

import sys
import h5py
import numpy as np


def walk(g, prefix=""):
    for k in g.keys():
        item = g[k]
        path = f"{prefix}/{k}"
        if hasattr(item, "shape"):
            print(f"  {path} shape={item.shape} dtype={item.dtype}")
        else:
            walk(item, path)


def find_logits(g, prefix=""):
    for k in g.keys():
        item = g[k]
        path = f"{prefix}/{k}"
        if hasattr(item, "shape"):
            if "logit" in k.lower():
                yield path, item
        else:
            yield from find_logits(item, path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r".\data\project_AidFOG\trial_force_cue_smoke\aidfog_ai.hdf5"
    print(f"Opening: {path}\n")
    with h5py.File(path, "r") as f:
        print("Structure:")
        walk(f)
        print()

        for lpath, dset in find_logits(f):
            arr = dset[...]
            print(f"Logits dataset: {lpath}  shape={arr.shape}  dtype={arr.dtype}")
            print(f"  raw range: min={arr.min():.3f}  max={arr.max():.3f}")

            # Squeeze and softmax over class axis (assume last dim is 2 classes).
            a = np.squeeze(arr).astype(np.float64)
            if a.ndim == 1 and a.shape[0] == 2:
                a = a[None, :]
            if a.ndim != 2 or a.shape[-1] != 2:
                print(f"  unexpected shape after squeeze: {a.shape} — skipping softmax")
                continue
            shifted = a - a.max(axis=-1, keepdims=True)
            ex = np.exp(shifted)
            probs = ex / ex.sum(axis=-1, keepdims=True)
            fog_prob = probs[:, 1]
            n = fog_prob.size
            print(f"  fog_prob: n={n}  mean={fog_prob.mean():.3f}  max={fog_prob.max():.3f}")
            for thr in (0.3, 0.5, 0.7):
                count = int((fog_prob >= thr).sum())
                pct = 100.0 * count / max(n, 1)
                print(f"    frames with fog_prob >= {thr}: {count}/{n} ({pct:.1f}%)")
            # Longest consecutive run above 0.3
            mask = fog_prob >= 0.3
            best = cur = 0
            for v in mask:
                cur = cur + 1 if v else 0
                if cur > best:
                    best = cur
            print(f"  longest run with fog_prob >= 0.3: {best} consecutive frames")

        # Ground-truth FoG label distribution from the replay producer.
        try:
            labels = f["aidfog_replay/dots-imu/fog_label"][...]
            labels = np.squeeze(labels).astype(np.int32)
            n = labels.size
            n_fog = int((labels == 1).sum())
            print(f"\nGround-truth fog_label: n={n}  fog_frames={n_fog} ({100.0*n_fog/max(n,1):.1f}%)")
            if n_fog > 0:
                # Longest consecutive FoG run in ground truth.
                best = cur = 0
                for v in labels == 1:
                    cur = cur + 1 if v else 0
                    if cur > best:
                        best = cur
                print(f"  longest GT FoG segment: {best} consecutive frames ({best/60:.2f} s @ 60 Hz)")
        except KeyError:
            print("\nNo fog_label dataset found.")


if __name__ == "__main__":
    main()
