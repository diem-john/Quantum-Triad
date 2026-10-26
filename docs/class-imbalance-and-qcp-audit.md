# Class Imbalance, Split Integrity, and QCP Audit

## What the pipeline currently does

- `Failure_Label` is generated as a synthetic vulnerability proxy using `label = 1 if total_risk >= 0.45 else 0`, where `total_risk` combines the vulnerability curve, coastal exposure, and accumulated fatigue.
- Snapshot extraction previously injected labels into four coastal buses if no failures were naturally generated. This could manufacture targets and obscure a label-generation/threshold problem. The fallback has been removed: when no positive labels are produced, the app stops and asks the user to inspect the risk rule and selected storms.
- Snapshot selection previously sampled randomly even after identifying failure-containing snapshots. The updated selector prioritizes failure-containing snapshots and then reserves a snapshot per storm, subject to the snapshot budget.
- Event-level partitions remain grouped by `storm_id` to prevent snapshots from the same storm leaking across partitions. The splitter now uses randomized multi-start whole-storm assignment followed by pair-swap refinement. Its global objective balances partition snapshot counts, positive-node counts, and failure-containing storm counts against target proportions of 60% train, 15% validation, 10% calibration, and 15% test. It requires at least one failure-containing storm per partition and stops rather than fabricating labels when this is infeasible. This is an allocation heuristic, not a guarantee of perfect class proportions; inspect the generated per-partition audit before training.
- A separate validation partition is used for model selection; calibration is reserved for QCP calibration rather than model selection.
- Validation uses the element-wise BCE criterion followed by a mean reduction before calling `.item()`, preventing the runtime error caused by attempting to convert a multi-element loss tensor to a scalar.
- The calibration UI warns against downsampling Safe nodes, because doing so changes the calibration distribution. The inference UI protects the untouched held-out test partition as the primary evaluation set. A balanced diagnostic test set may be reported only as a separate secondary analysis; it must not replace the original test metrics.
- The QSTGNN training loss now weights positive-class node errors using the class ratio calculated from the training partition only. No validation, calibration, or test examples are resampled.\n- In the A+B feature encoder, branch B fidelity similarities are multiplied by a configurable `b_weight` before concatenation. The default is `0.25`; branch A expectation values are unchanged. Set the weight to `0` to ablate B, or increase it for a sensitivity experiment. The encoder persists the weight with its train-only landmarks so inference uses the same transformation.\n- The Streamlit QKN Benchmark tab reports A-only expectation-feature, B-only fidelity-kernel, A+B weighted-fusion, and classical RBF-SVC results on a shared event-level holdout. The benchmark is a row-wise SVC experiment and should not be presented as a full QSTGNN comparison.
- The app exports and displays pre-selection label counts and counts for the full selected dataset, training, validation, calibration, and testing. Counts include Safe, Failure, percentages, total samples, and unexpected labels.
- Phase 5 computes raw and temperature-scaled probabilities on validation and held-out test partitions. Thresholds are selected on validation only using either F2 maximization or maximum precision subject to a configurable minimum-recall constraint; if the constraint is infeasible, the app reports that and falls back to the highest-recall validation threshold. The selected thresholds are frozen before held-out test metrics are computed; test labels are not used for threshold selection.
- The extraction/split workflow displays and exports a per-storm failure-prevalence audit (snapshot count, bus-label count, failure count, and failure percentage) to help diagnose class-rate differences across event-level partitions. Do not rebalance the test set or move individual snapshots across partitions to make distributions look more similar.
- Phase 5 reports decision thresholds beside held-out test metrics. QCP fail-safe remains a separate uncertainty-based decision rule and is not tuned with the binary validation threshold. Report its coverage and ambiguity separately from binary classification metrics.
- Phase 5 reports empirical conformal set coverage separately from the ambiguous-set rate and fail-safe binary metrics.

## Threshold policy and probability-quality diagnostics

Phase 5 supports two validation-only threshold policies:

1. **Maximize F2 (recall-weighted):** select the threshold with the highest validation F2; precision and threshold provide deterministic tie-breaks.
2. **Maximize precision subject to minimum recall:** select the highest-precision validation threshold whose recall meets the user-selected target. If no candidate reaches the target, the UI explicitly reports this and falls back to the highest-recall validation threshold.

The chosen policy is applied independently to raw and temperature-scaled validation probabilities. Those thresholds are then frozen and applied to the held-out test probabilities. The test labels are not used to select thresholds.

Phase 5 also reports Brier score and log loss for raw and temperature-scaled probabilities on validation and held-out test data, plus held-out probability-distribution plots, a reliability diagram, and a downloadable reliability-bin table. Lower Brier score and log loss are better. The test-set diagnostics are reporting-only; do not use them to tune temperature, thresholds, features, or model settings. Inspect probability distributions and reliability alongside Recall, Precision, F1, and Accuracy: thresholded metrics alone do not establish that predicted probabilities are calibrated.

The per-storm audit reports storm ID, snapshot count, bus-label count, Failure-label count, and Failure-label prevalence. Review it before training to understand whether partition-level prevalence differences are driven by a small number of storms. Do not move snapshots between partitions or split a storm across partitions to force matching class ratios.

## Target interpretation

`Failure_Label` is generated by the configured synthetic vulnerability rule (`total_risk >= 0.45`); it is not a record of an observed real-world grid failure. Therefore, all reported classification and calibration metrics describe agreement with this synthetic proxy. Claims about actual grid-failure prediction require validation against observed outage/failure labels.

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
6. Report QCP empirical set coverage, ambiguity rate, and fail-safe binary metrics separately.\n7. For fusion sensitivity, compare B weights (for example 0, 0.10, 0.25, 0.50, 1.0) on identical splits. Choose weights using validation data only; keep the held-out test set for the final single comparison.
7. Re-run training and inference locally to populate actual metrics. A repository commit cannot execute against the user's local source CSVs, model runtime, or environment.
