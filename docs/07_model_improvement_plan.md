# EEG Model Improvement Protocol

Date: 2026-10-01. This stage builds on the existing fatigue EEG baseline. Model quality
and evaluation remain the priority; Agents and IoT integration follow later.

## Problems To Address

- The dataset has only 12 subjects and recording-level normal/fatigue labels.
- Subject 06 has a severely contaminated FT7 electrode. Common average reference can
  spread its artifact to otherwise useful channels.
- The v2 model has a large subject-dependent score offset. Its validation decision
  threshold of about 0.00029 does not support transfer to an unseen person.
- Two validation subjects cannot support broad hyperparameter exploration. Previously
  inspected test results are historical results, not a fresh independent test.

## Implementation And Evaluation

1. Preserve original raw data, baseline features, checkpoint and reports.
2. Generate a separately versioned feature dataset using per-window quality checks,
   robust reference and channel repair, with the same documented channel order and
   frequency-band contract. A prediction rejected for poor quality is not a normal
   prediction, and rejection coverage must be reported with accuracy.
3. Compare a small declared candidate set: relative channel bandpower, anatomical
   regional aggregation, band ratios and five-band entropy; regularized Logistic
   Regression, shrinkage LDA and RBF SVC. Scaling must be fitted on each training fold.
4. Use subject-group cross-validation within training subjects for model comparison
   and probability calibration. No window-random split and no use of another
   recording of the held subject in training.
5. Freeze the candidate, preprocessing, calibration and decision rule before looking
   at evaluation subjects. Report overall and per-subject AUROC, balanced accuracy,
   macro F1, sensitivity, specificity, Brier score, confusion matrices and coverage.
6. Produce a usable checkpoint, NPZ/CNT inference commands, prediction CSVs, metrics,
   source hashes and a reproducible report. Keep the old model for comparison.

Group-level calibration and selection require care. A Platt calibrator fitted on all
training out-of-fold scores may be useful for the final model, but measuring that
calibrator on the same scores is an in-sample calibration result. Honest validation
must fit the calibrator inside the outer training fold. A fixed 0.5 decision rule can
be compared with a threshold selected using only inner training folds. An extreme
threshold selected on two validation subjects should not become the default merely
because it gives the best apparent validation accuracy.

## Development Experiment

The local experiment `runs/quality_validation_experiment.py` reads only training
subjects 03, 04, 06, 07, 08, 09, 10 and 12, plus validation subjects 01 and 05. It does
not open the NPZ files for subjects 02 or 11. It declares 24 feature/classifier
combinations and compares three training-only calibration implementations:

- One Platt model fitted on pooled subject-held-out raw scores, applied to a model
  fitted on all training subjects.
- The same pooled Platt model applied to an ensemble of subject-held-out estimators.
- Scikit-learn sigmoid calibration inside each training held-subject fold, followed
  by the mean of the calibrated fold predictions.

Regions follow the documented channel order: frontal FP1/FP2/F7/F3/FZ/F4/F8/FT7/
FC3/FCZ/FC4/FT8, central C3/CZ/C4, parietal CP3/CPZ/CP4/P3/PZ/P4, and occipital
O1/OZ/O2. Each region uses the median. Lateral temporal channels are excluded from
this reduced representation. Ratios are theta/beta, alpha/beta, delta/beta and
theta/alpha in log space. Entropy is the normalized entropy of the five integrated
band powers; it is not broadband spectral entropy.

Riemannian covariance methods need raw covariance features. Those cannot be
reconstructed from 150 log-bandpower values. Adding them would be a separate declared
feature extraction experiment, using a maintained library such as pyRiemann.

## Frozen v3 Protocol

The exploratory train/validation results support retaining all 150 relative bandpower
features. The final benchmark declares only three candidates: shrinkage LDA (`auto`),
Logistic Regression C=0.1, and Logistic Regression C=1. Each uses three subject-disjoint
classifier/calibration members and averages their calibrated probabilities.

Every outer LOSO fold performs inner LOSO selection by mean subject log loss, then
balanced accuracy at a fixed 0.5 threshold. Scaling and base classifier fitting exclude
both the calibration subjects and the outer evaluation subject. Calibration excludes
the outer evaluation subject. No threshold is chosen from outer labels.

Deployment selection uses full-cohort inner LOSO before any outer benchmark report is
inspected. Its exported model uses all development subjects and has no independent
held-out score. The original legacy preprocessing and robust preprocessing are evaluated
with identical candidate grids and rules. Their paired subject bootstrap includes
coverage-adjusted balanced accuracy, counting each abstention as incorrect.

This is an adaptive development study on the original 12-person cohort, including
subjects whose historical baseline test results have already been inspected. Nested
LOSO prevents fit-time subject leakage; it cannot turn this cohort into a new external
test set or remove uncertainty from the earlier research choices.

## Next Data Collection

An independently collected or licensed dataset is still needed to establish external
validity. For EEG-camera fusion, record both modalities on the same person with
aligned timestamps and event annotations. An optional personal calibration period
requires a separate evaluation protocol and must not be described as zero-shot
prediction. The current recording-level fatigue labels cannot establish microsleep
onset, detection delay or real-road false alerts per hour.
