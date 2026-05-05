"""
Canonical F1 / sample-metric / aggregation functions — verbatim from Alex.

Vendored 2026-04-30 (post-Vayalet 2026-04-29 §3c) so the cueing-side analysis
produces F1 numbers from the same code path as the model-side analysis.
**Do not modify** without coordinating with Alex; the whole point is internal
methodological consistency between the two halves of the joint paper.

Two distinct F1 flavours are implemented:

  - `segment_f1_score(y_pred, y_true, overlap=0.5)` — *event-based* F1.
    Builds run-length segments from binary streams, label-aware, and treats a
    predicted segment as TP if its IoU with the *closest* ground-truth segment
    of the same label is ≥ `overlap` AND that GT segment hasn't already been
    matched. Otherwise FP. Unmatched GT segments → FN.

    Lineage (Alex confirmed 2026-05-04):
      * Inherited from Juha's original `compute_segment_f1` and ported as-is
        for pipeline consistency. The both-label behaviour was not a conscious
        design choice — it was simply part of Juha's implementation.
      * For the published reference of IoU-based segment matching itself, cite
        Lea et al. 2016 (action segmentation) — the methodological lineage
        this implementation comes from.

    Methodological notes — non-obvious:
      * `segments_from_binary` produces segments for BOTH labels (0 and 1) —
        not just the positive class. Combined with the label-aware IoU mask
        (`pl == y_labels`), each predicted-0 run is matched against GT-0
        runs and each predicted-1 run against GT-1 runs. Both are counted
        into the same TP/FP/FN tallies.
      * Practical impact (Alex's note 2026-05-04): in our trials the no-fog
        stretches are much longer and more frequent than FoG episodes, so
        they likely inflate F1 somewhat versus a positive-class-only score.
        Document this trade-off when reporting numbers; if reviewers push
        back, an easy fix is to filter to class 1 only before matching.
      * Greedy first-match by argmax of IoU; not bipartite-optimal.
      * `overlap=0.5` is Alex's IoU≥50 convention.

  - `compute_sample_metrics(y_true, y_pred)` — *sample-wise* F1 via sklearn
    confusion matrix on the concatenated frame stream.

`compute_metrics` is the per-subject + global aggregation harness used by
Alex's evaluation pipeline. Trial F1s are rounded to 2 d.p. *before* averaging
(intentional, matches his published numbers).

Dependency note: `compute_sample_metrics` requires `sklearn.metrics.confusion_matrix`.
"""

from __future__ import annotations

import numpy as np

# `sklearn` is imported lazily inside `compute_sample_metrics` so that callers
# only needing the segment-based F1 path (`segment_f1_score`) don't need the
# extra dependency. Alex's original module imports it at module level.


def segments_from_binary(y):
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        return []
    labels_out, starts, ends = [], [], []
    last = y[0]
    labels_out.append(int(last))
    starts.append(0)
    for i in range(len(y)):
        if y[i] != last:
            labels_out.append(int(y[i]))
            starts.append(i)
            ends.append(i)
            last = y[i]
    ends.append(len(y))
    return list(zip(starts, ends, labels_out))


def segment_f1_score(y_pred, y_true, overlap=0.5):
    p_segs = segments_from_binary(y_pred)
    y_segs = segments_from_binary(y_true)

    tp, fp = 0, 0
    hits   = np.zeros(len(y_segs), dtype=int)

    if len(y_segs) == 0:
        fp = len(p_segs)
    else:
        y_starts = np.array([s for s, _, _ in y_segs])
        y_ends   = np.array([e for _, e, _ in y_segs])
        y_labels = np.array([l for _, _, l in y_segs])

        for ps, pe, pl in p_segs:
            intersection = np.maximum(0, np.minimum(pe, y_ends) - np.maximum(ps, y_starts))
            union        = np.maximum(pe, y_ends) - np.minimum(ps, y_starts)
            iou          = (intersection / union) * (pl == y_labels)

            idx = int(np.argmax(iou))
            if iou[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1

    fn        = len(y_segs) - int(hits.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return f1, tp, fp, fn


def compute_sample_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix  # lazy: keeps segment_f1_score importable without sklearn
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    fp_rate   = FP / (FP + TN + 1e-9)
    return {
        'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
        'precision': precision, 'recall': recall,
        'f1': f1, 'accuracy': accuracy, 'fp_rate': fp_rate,
    }


def compute_metrics(walk_cache):
    subjects = {}
    for walk_id, labels in walk_cache.items():
        subjects.setdefault(walk_id.split('-')[0], []).append(labels)

    per_subject        = {}
    all_trial_f1s      = []
    all_trial_fp_rates = []

    for subj in sorted(subjects):
        walks = subjects[subj]

        subj_true, subj_pred   = [], []
        subj_trial_f1s         = []
        subj_trial_fp_rates    = []

        for labels in walks:
            subj_true.append(labels['true'])
            subj_pred.append(labels['pred'])
            f1, tp, fp, fn = segment_f1_score(labels['pred'], labels['true'])
            subj_trial_f1s.append(round(float(f1), 2))
            y_inv = 1 - np.asarray(labels['true']).astype(int)
            p_inv = 1 - np.asarray(labels['pred']).astype(int)
            _, tn, _, _ = segment_f1_score(p_inv, y_inv)
            subj_trial_fp_rates.append(fp / (fp + tn + 1e-9))

        per_subject[subj] = {
            'sample':        compute_sample_metrics(
                                 np.concatenate(subj_true),
                                 np.concatenate(subj_pred)),
            'event_f1':      round(float(np.mean(subj_trial_f1s)), 2),
            'event_fp_rate': round(float(np.mean(subj_trial_fp_rates)), 4),
            'trial_f1s':     subj_trial_f1s,
        }
        all_trial_f1s.extend(subj_trial_f1s)
        all_trial_fp_rates.extend(subj_trial_fp_rates)

    metric_keys = ['f1', 'precision', 'recall', 'accuracy', 'fp_rate']
    macro_sample = {
        k: round(float(np.mean([per_subject[s]['sample'][k] for s in per_subject])), 4)
        for k in metric_keys
    }
    macro_event_f1      = round(float(np.mean(all_trial_f1s)), 4)
    macro_event_fp_rate = round(float(np.mean(all_trial_fp_rates)), 4)

    global_metrics = {
        'sample':        macro_sample,
        'event_f1':      macro_event_f1,
        'event_fp_rate': macro_event_fp_rate,
        'trial_f1s':     all_trial_f1s,
    }

    return per_subject, global_metrics
