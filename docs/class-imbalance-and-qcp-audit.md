# Class Imbalance, Split Integrity, and QCP Audit

## What the pipeline currently does

- `Failure_Label` is generated as a synthetic vulnerability proxy using `label = 1 if total_risk >= 0.45 else 0`, where `total_risk` combines the vulnerability curve, coastal exposure, and accumulated fatigue.
- Snapshot extraction previously injected labels into four coastal buses if no failures were naturally generated. This could manufacture targets and obscure a label-generation/threshold problem. The fallback has been removed: when no positive labels are produced, the app stops and asks the user to inspect the risk rule and selected storms.
- Snapshot selection previously sampled randomly even after identifying failure-containing snapshots. The updated selector prioritizes failure-containing snapshots and then reserves a snapshot per storm, subject to the snapshot budget.
- Event-level partitions remain grouped by `storm_id` to prevent snapshots from the same storm leaking across partitions. The latest splitter searches whole-storm allocations against target proportions of 60% train, 15% validation, 10% calibration, and 15% test while balancing snapshot counts and positive-node burden. It requires at least one failure-containing storm per partition and stops rather than fabricating labels when this is infeasible.
- A separate validation partition is used for model selection; calibration is reserved for QCP calibration rather than model selection.
- Validation uses the element-wise BCE criterion followed by a mean reduction before calling `.item()`, preventing the runtime error caused by attempting to convert a multi-element loss tensor to a scalar.
- The calibration UI warns against downsampling Safe nodes, because doing so changes the calibration distribution. The inference UI protects the untouched held-out test partition as the primary evaluation set. A balanced diagnostic test set may be reported only as a separate secondary analysis; it must not replace the original test metrics.
- The QSTGNN training loss now weights positive-class node errors using the class ratio calculated from the training partition only. No validation, calibration, or test examples are resampled.
- The app exports and displays pre-selection label counts and counts for the full selected dataset, training, validation, calibration, and testing. Counts include Safe, Failure, percentages, total samples, and unexpected labels.
- Phase 5 reports empirical conformal set coverage separately from the ambiguous-set rate and fail-safe binary metrics.

## Important limitation: SMOTE and graph tensors

The QSTGNN input is a graph snapshot with shape `(snapshots, 33 buses, time steps, channels)`, and its target is a multi-node label matrix `(snapshots, 33 buses)`. Standard `imblearn.SMOTE` expects independent tabular feature rows with one class label per row. Flattening graph snapshots and applying SMOTE naively would destroy graph/node semantics or create labels that do not correspond to the generated graph state.

Therefore, standard SMOTE has **not** been applied to the QSTGNN graph tensors: doing so directly would break graph/node semantics. The current implementation uses a training-only positive-class weighted loss as the graph-preserving baseline. A graph-aware SMOTE method should be a separate experiment only after its augmentation unit, topology preservation, and label assignment are explicitly defined. Any future resampling must fit on training data only; validation, calibration, and test data must remain untouched.

## Split feasibility and interpretation

The app requires at least four distinct storm events to form train, validation, calibration, and test partitions, but four events alone does not guarantee that both classes will appear in every partition. Inspect the generated distribution report. If a partition lacks Failure labels, its Failure Recall/Precision/F1 cannot establish positive-class detection performance. Increase the number of storms or revise the event-level allocation; do not manufacture positive labels or split snapshots from one storm across partitions.

## Required controlled comparison

1. Generate and review label distributions before snapshot selection and after each split.
2. Validate the `total_risk >= 0.45` label rule against the underlying score distribution.
3. Keep identical storm-level train/validation/calibration/test indices for A-only and A+B.
4. Fit any resampling method using training data only.
5. Compare A-only and A+B using Recall first, then Precision, F1, and Accuracy, with identical test labels and decision thresholds.
6. Report QCP empirical set coverage, ambiguity rate, and fail-safe binary metrics separately.
7. Re-run training and inference locally to populate actual metrics. A repository commit cannot execute against the user's local source CSVs, model runtime, or environment.
