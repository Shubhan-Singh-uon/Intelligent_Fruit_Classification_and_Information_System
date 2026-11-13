<!--
File: model_card.md
Project: Intelligent Fruit Classification and Information System
Description: Documents the selected model, intended use, inference
requirements, evaluation results, and known limitations.
Author: Shubhan Singh
Responsible Pillar: Pillar 2 - Machine Learning & Evaluation
-->
# Model Card: Apple vs Banana Classifier (M4, EfficientNetB0 Fine-Tuned)

**Project:** Intelligent Fruit Classification and Information System (COMP6900, Group 8)

**Deployed run:** `M4_PC1`, version string `M4_PC1-d7e2e21cf167`


This card follows the structure proposed by Mitchell et al. (2019). It describes the model that the application loads, the evidence for it, and its limits. Where this card quotes a configuration value, `src/fruit_classifier/config.py` is the authoritative source.

## 1. Model Details

| Item | Detail |
|------|--------|
| Model | Binary image classifier, class 0 = Apple, class 1 = Banana |
| Experiment ID | M4, "EfficientNetB0 Fine-Tuned" |
| Deployed run | `M4_PC1` (checkpoint `artifacts/M4_PC1.pt`) |
| Date | 20 October 2026 (deployment record) |
| Developers | Pillar 2: Model development and evaluation by  (Shubhan Singh) |
| Architecture | EfficientNetB0 (Tan and Le, 2019) from torchvision, with a new head: dropout 0.30, then a linear layer from 1,280 features to 2 classes |
| Parameters | 4,010,110; checkpoint 15.57 MiB |
| Input | RGB image resized to 128 × 128, pixel values in 0–1 (preprocessing `PP_v1`); ImageNet mean and standard deviation are applied inside the network |
| Output | Probabilities for Apple and Banana, plus the 1,280-value feature vector used for the distance signal (Section 9) |
| Initialisation | ImageNet-1k pretrained weights (`IMAGENET1K_V1`), then M3 (head trained on frozen features), then fine-tuning |
| Framework | PyTorch 2.7.1 (CPU), Windows 11, for the deployed run |
| Configuration hash | `d7e2e21cf167` (identical on all four computers) |
| Dataset | `Apple_Banana_Dataset_(Final)`, fingerprint `52b395cdef84b0ff`; see the [dataset card](dataset_card.md) |
| Terms of use | Academic, non-commercial; see [data_sources_and_licensing.md](data_sources_and_licensing.md) (Section 4) |

## 2. Intended Use

**Primary use.** To identify whether a picture shows an apple or a banana, as the classification stage of an information system that then shows nutrition, health, usage and general facts about the identified fruit. The model is one component of an academic prototype; the application also decides whether a prediction can be trusted (Section 9).

**Intended users.** Students, markers and other readers of the COMP6900 project.

**Out-of-scope uses.** The model must not be used to:

- judge ripeness, quality, freshness, spoilage or food safety (it was never trained or evaluated for these);
- identify any fruit other than apples and bananas (a different fruit is not recognised as such);
- support decisions about health, diet, allergies or purchasing;
- operate in a commercial or safety-relevant setting.

## 3. Factors

**Relevant Factors.** Accuracy may depend on the image source (public or student-taken photographs), lighting, background, camera angle and distance, the number of fruit in the picture, the fruit variety and state, and whether the image shows an apple or banana at all.

**Evaluation Factors.** The evaluation reports results separately for public and student-taken test images, for 100 unsupported images in five categories (Food, Other fruit, Ambiguous, Background, Objects), for a 90° rotation, and for increasing amounts of pixel noise. It does not report results by fruit variety, lighting or background, because these were not recorded.

## 4. Metrics

| Metric | What it measures |
|--------|------------------|
| Accuracy, precision, recall, F1 | Correctness of the Apple/Banana decision on the held-out test set |
| Confusion matrix | Distribution of correct and incorrect classifications |
| Confidence (highest class probability) | How strongly the model favours its choice |
| Confidently wrong predictions | Test images misclassified with confidence at or above 0.90 |
| ROC AUC of confidence | How well confidence separates correct from incorrect predictions |
| Feature distance | Mahalanobis distance from an image's features to the nearest class centroid, using one shared covariance (Lee et al., 2018; Ledoit and Wolf, 2004); large values mean the image resembles none of the training images |
| Stability under test-time views | Whether the predicted class is unchanged under a horizontal flip, rotations of ±10° and a brightness increase |
| Unsupported images wrongly accepted | Share of out-of-distribution images that would still be accepted as an apple or banana |
| Outcome split | Share of images the policy would Accept, send to Review or Reject (Section 9) |
| Parameters, checkpoint size, CPU latency | Computational cost |

## 5. Evaluation Data

| Data | Size | Notes |
|------|-----:|-------|
| Test set | 382 | 192 Apple and 190 Banana; 191 public and 191 student-taken. Not used for training or for early stopping (the validation set was); each model was evaluated on it once. The deployment thresholds in Section 9 were, however, calibrated on its feature distances (see Section 9.4). |
| Unsupported (OOD) set | 100 | Food 28, Other fruit 24, Ambiguous 16, Background 16, Objects 16; 50 public and 50 student-taken. Separate from the 2,544 dataset images. Evaluated on two of the four computers (PC1 and PC2). |
| Robustness and stability samples | 60 | Random test images used for the stability, 90° rotation and noise checks. |

## 6. Training Data and Procedure

**Data.** 1,780 training images (890 per class). The 382 validation images were used for early stopping only.

**Two-stage training of M4.** M3 trains the new head on frozen pretrained features. M4 starts from M3, unfreezes the later part of the network (`unfreeze_from = 6`: feature blocks 6 to 8 and the head), and continues at a ten times smaller learning rate. Warm-starting from M3 avoids large gradients from a randomly initialised head damaging the pretrained layers.

| Setting | M4 |
|---------|----|
| Optimiser | AdamW (Loshchilov and Hutter, 2019), learning rate 1e-4, weight decay 1e-4 |
| Schedule | Cosine decay |
| Loss | Cross-entropy without label smoothing (so that probabilities can be interpreted directly) |
| Maximum epochs, patience | 15 epochs; early stopping after 5 epochs without improvement (minimum improvement 0.001) |
| Batch size, seed | 32; 42 |
| Augmentation (training only) | Horizontal flip (p = 0.5), rotation ±15°, random zoom 85–100% of the area, brightness and contrast ±0.2, saturation ±0.1; no hue shift, vertical flip or noise |
| Hardware | CPU only |

Colour is the strongest cue between the two classes, so hue was deliberately not altered. All models in the experiment were trained and evaluated with identical data, seed and configuration on four computers (16 runs in total).

## 7. Quantitative Results

### 7.1 Comparison of the Four Candidate Models

Values are means over four runs, one per computer. The experiment tests reproducibility across computing environments, not four independent training samples.

| Model | Description | Test accuracy (range) | Confidently wrong per run | Student–public accuracy gap | Parameters | CPU ms per image | OOD wrongly accepted |
|-------|-------------|-----------------------|--------------------------:|----------------------------:|-----------:|-----------------:|---------------------:|
| M1 | Custom CNN baseline (four convolutional blocks, trained from scratch) | 0.9594 (0.9581–0.9607) | 1.75 | 0.0079 | 389,890 | 8.4 | 7.5% |
| M2 | Custom CNN, tuned (two convolutions per block, stronger regularisation and augmentation, trained from scratch) | 0.9843 (0.9817–0.9869) | 0 | 0.0314 | 1,174,690 | 18.5 | 13.0% |
| M3 | EfficientNetB0 transfer learning (frozen pretrained backbone, new head) | 0.9974 (0.9974) | 0 | 0.0052 | 4,010,110 | 24.2 | 0.0% |
| **M4** | **EfficientNetB0, fine-tuned** | **1.0000 (1.0000)** | **0** | **0.0000** | **4,010,110** | **22.3** | **0.0%** |

Notes: macro F1 equals accuracy for all four models. The "OOD wrongly accepted" column uses the original notebook criterion (confidence ≥ 0.90, feature distance within the 95th percentile of training distances, stable predictions) on PC1 and PC2 only; Section 9 explains why that criterion is too strict to be used as it stands. Latency is a mean of 50 single-image runs on each computer's CPU.

The improvement from M1 to M2 (+0.0249), from M2 to M3 (+0.0131) and from M3 to M4 (+0.0026) had the same direction on every computer.

### 7.2 Selected Model on the Test Set

| Result | Value |
|--------|-------|
| Correct predictions | 382 of 382 (all four runs) |
| Accuracy, precision, recall and F1 | 1.000 for both classes |
| Accuracy on public / student images | 1.000 (191 of 191) / 1.000 (191 of 191) |
| Mean confidence on correct predictions | 0.999; every test image scored at least 0.90, and about 89% scored at least 0.999 |
| Exact 95% lower bound for accuracy | 99.04% (Clopper–Pearson, 382 of 382 correct) |

A perfect score on 382 images shows that the error rate is very likely below 1%, not that it is zero. Section 11 lists further reasons for caution.

### 7.3 Training Behaviour

M4 stopped after 12 epochs, with the best validation result at epoch 7 in all four runs. The final gap between training and validation accuracy was 0.0026, and the final validation loss was 0.0059 to 0.0062. There is no sign of overfitting in the recorded histories. M3 reached its best epoch at 10 of 15.

### 7.4 Reproducibility across Computers

All four computers used the same dataset fingerprint and configuration hash. Test accuracy was identical (M4, M3) or within 0.0052 (M2) across them. The environments differed in operating system (Windows 10 and 11), CPU, memory and PyTorch version (2.7.1 on PC1; 2.14.1 on PC2 to PC4). Because the code, data and seed were identical, agreement across computers shows that the result is reproducible in this setting; it is not independent evidence from four separate training samples.

### 7.5 Robustness (60 test images)

| Check | Result for M4 |
|-------|---------------|
| Rotation by 90° | Accuracy 100% before and after |
| Prediction changes under the test-time views | None (no sampled test image was flagged as unstable) |
| Random pixels replacing 10% / 30% / 50% / 70% / 90% of an image | Mean confidence 0.987 / 0.911 / 0.754 / 0.652 / 0.568 |

Confidence falls steadily as the corruption increases, which is the desired behaviour: the model does not stay confident on images it can no longer read.

### 7.6 Computational Cost

4,010,110 parameters, a 15.57 MiB checkpoint, and about 22 ms per image on a CPU for one pass through the network (mean of 50 runs on each of four computers). A complete classification evaluates the network five times (the image and the four stability views) and therefore takes roughly five times as long. Training M4 took about 450 s (mean of the four computers, 190 s to 615 s).

## 8. Model Selection Process

The collation notebook applies one explicit, auditable rule rather than choosing the highest accuracy automatically. The rule is applied to the evaluation results, so no model was tuned on the test set during selection.

| Step | Rule | Outcome |
|------|------|---------|
| 1 | Eliminate a model whose confidently-wrong rate is more than twice the median of the other models (or above all of them, when that median is zero) | M1 eliminated (1.75 per run against 0 for each other model) |
| 2 | Eliminate a model whose public-versus-student accuracy gap exceeds 0.05, unless that would eliminate every model | No model eliminated (largest gap 0.0314) |
| 3 | Eliminate a model whose rate of wrongly accepting unsupported images is a clear outlier, by the same comparison as Step 1 | M2 eliminated (13.0% against 7.5%, 0% and 0% for the others) |
| 4 | Flag, but do not eliminate for, an accuracy spread across computers above 0.03 | No model flagged (largest spread 0.0052) |
| 5 | Among the remaining models rank by accuracy. A difference smaller than 0.005 is not treated as a tie if the leader wins on every paired computer; a genuine tie falls back to the model with the fewest parameters | M4 over M3: 1.0000 against 0.9974, ahead on all four computers |

**Interpretation.** M4 and M3 differ by one test image (1 of 382, 0.0026). The margin is the same on all four computers, but those runs share data and seed, so the consistency demonstrates reproducibility and not statistical significance. M4 was chosen under the rule above; M3 would be a defensible alternative. The OOD criterion used in Step 3 was available for only two of the four computers.

**Choice of run.** The notebook sorts the four M4 runs by accuracy and takes the first row. All four tie at 1.0000, so `M4_PC1` is the first run in table order, not the winner of a comparison. An independent comparison of the four checkpoints for this card found them functionally equivalent:

| Comparison | Finding |
|------------|---------|
| Files | Four distinct files; `M4_PC1_1.pt`, a fifth file, is a byte-identical copy of `M4_PC1.pt` |
| Weights | `M4_PC2`, `M4_PC3` and `M4_PC4` agree with one another to floating-point precision (relative distance about 5e-7). `M4_PC1` differs from each of them by about 9e-4, only in the layers that were fine-tuned; the frozen layers are bit-identical |
| Likely cause | PyTorch 2.7.1 on PC1 against 2.14.1 elsewhere (the three newer-version runs agree despite different CPUs and thread counts) |
| Behaviour | Identical predictions on every spot-check image; probabilities differ by at most 0.25 percentage points; test-set feature distances have median 41.19 (`M4_PC1`) and 41.30 (the others) |
| Validation loss | 0.0062 (`M4_PC1`) against 0.0059 (others), a negligible difference |

`M4_PC1` is retained because the deployment record, the calibrated statistics and the thresholds in this card all refer to it. If another run is deployed, its own statistics and thresholds must be generated (see [artifacts/README.md](../artifacts/README.md)).

## 9. Decision Policy and Calibration

### 9.1 Why accuracy alone is not enough

A two-class classifier must always answer Apple or Banana, even for a photograph of a mountain. Its confidence is not a reliable guard: in an informal check, three cropped screenshots of a user interface (clearly not fruit) were each classified as Apple with 91% to 96% confidence. The application therefore combines three signals (confidence, feature distance, stability) and assigns each upload one of four outcomes.

### 9.2 Outcomes

| Outcome | Rule (checked in this order) | Application behaviour |
|---------|------------------------------|-----------------------|
| Rejected | Feature distance above the limit | No fruit is confirmed and no fruit information is shown |
| Needs review | Confidence below the threshold, or the prediction changes under the test-time views | Tentative prediction and probabilities, with an uncertainty warning; no fruit information |
| Accepted | Otherwise | Fruit, both probabilities, confidence and database information are shown |
| Invalid input | The file cannot be read as a supported image | Error message; the model is never run |

Distance is an out-of-distribution signal; it is not proof that an image contains no fruit.

### 9.3 Operating Point

At the time of writing `config.py` sets the confidence threshold to **0.90** and the feature-distance limit to **61.69**. The notebook's original distance cutoff, the 95th percentile of the *training* distances (35.73), kept only 31.9% of genuine test images, because training images lie closer to the class centres than unseen ones; it was therefore replaced. The value 61.69 is the smallest limit that keeps 95% of the genuine test images, which was the agreed calibration target.

### 9.4 Calibration Evidence

The table shows how the policy would divide the 382 genuine test images and the 100 unsupported images for several settings. It was produced by `src/scripts/export_ood_stats.py` from the embeddings of `M4_PC1`.

**Methodological note.** The distance limits in this table were derived from the feature distances of the test images themselves (for example, 61.69 is the distance that keeps 95% of them), so the percentages for genuine images are *in-sample* for this calibration: they describe how the thresholds behave on the images used to set them. They do not affect the accuracy figures in Section 7, which depend only on the predicted class, but they are likely to be slightly optimistic for new photographs. A fresh set of genuine images would give an unbiased estimate (Section 11, item 5).

| Distance limit | Confidence threshold | Genuine: accepted / review / rejected | Unsupported images wrongly accepted |
|---------------:|---------------------:|---------------------------------------|------------------------------------:|
| **61.69 (current)** | **0.90 (current)** | **94.8% / 0.0% / 5.2%** | **19.0%** |
| 65 | 0.99 | 93.7% / 2.6% / 3.7% | 13.0% |
| 65 | 0.999 | 86.4% / 9.9% / 3.7% | 2.0% |
| 70 | 0.99 | 95.8% / 2.9% / 1.3% | 16.0% |
| 70 | 0.999 | 88.2% / 10.5% / 1.3% | 3.0% |
| 70 | 0.9995 | 84.8% / 13.9% / 1.3% | 0.0% |

**Reading the table.**

- Retaining 95% of genuine images does not by itself reject enough unsupported images: at the current setting 19% of the 100 unsupported images are still accepted, and 33% of the "Other fruit" images.
- The confidence threshold is the stronger lever. The model is extremely confident on genuine images (about 89% score 0.999 or more) and rarely so on unsupported ones, so raising the threshold from 0.90 to 0.999 lowers wrongly accepted unsupported images from about 16–19% to 2–3%. Genuine images below the threshold are sent to *Needs review* and not rejected, at a cost of roughly 10% of genuine images.
- The unsupported set is small (100 images) and the settings were examined on the same images, so differences of a few percentage points are within noise.

### 9.5 Informal Integration Checks

Five photographs, chosen by hand, were classified through the full application with the operating point in Section 9.3. These are anecdotal checks, not an evaluation.

| Photograph | Outcome | Observation |
|------------|---------|-------------|
| Two bananas on a table (an iPhone photo saved as `.png`) | Accepted | Banana, 100.0%; banana information displayed |
| Slices of watermelon | Needs review | Possible match Apple, 59.6%: uncertain, no fruit information |
| A mountain landscape | Rejected | Distance 62.8 against the limit 61.69 |
| A market pile of apples | Rejected | Classified as Apple with very high confidence, but the whole-pile photograph lies far from the single-fruit training images |
| A street drawing of a banana | Rejected | Banana 98.2%, but distance 69.6 |

The pile of apples shows the cost of a single distance limit: a valid photograph that looks unlike the training images is rejected. See Section 11.

## 10. Ethical Considerations

- **Data.** The images show fruit and no personal data is intentionally included; see [data_sources_and_licensing.md](data_sources_and_licensing.md) for the open privacy point on student-taken photographs. The application does not store uploaded images, only prediction metadata.
- **Misplaced trust.** A confident answer can still be wrong. The application shows probabilities, warns about uncertain predictions and refuses to confirm images that look unlike the training data, and its displayed health information is labelled as general information, not medical advice.
- **Scope.** The model identifies two fruit. A user should not read a result as a statement about ripeness, safety, allergens or nutritional need.
- **Environment.** Training used laptop CPUs for a few minutes per run; the environmental cost is small.
- **Fairness.** The task does not involve people, so fairness across demographic groups does not apply. Differences between public and student-taken photographs were checked (Section 7.2).

## 11. Caveats and (Recommendations for Future Work)

1. **The test result may be optimistic.** The public images include flipped, rotated and translated variants of other photographs, and images were assigned to splits individually. The audit summarised in the training notebook reports that none cross split boundaries, but this was not reproduced in the repository's preparation notebook (see [dataset_card.md](dataset_card.md), Section 7). *Recommendation:* group each photograph with its variants, confirm that every group lies in one split, and re-run the evaluation if any do not.
2. **The test set is small.** A result of 382 of 382 means an accuracy above about 99.0% with 95% confidence; M3 and M4 differ by one image.
3. **Closed set.** The model knows two classes. Other fruit (for example watermelon) can be classified confidently as Apple or Banana; only the distance and confidence signals prevent them from being accepted.
4. **A single distance limit trades two errors.** It rejects some valid but unusual photographs (such as a pile of fruit) and passes some unsupported images that happen to resemble the training data. The current operating point (Section 9.3) accepts 19% of the unsupported test images. *Recommendation:* raise `CONFIDENCE_THRESHOLD` (for example to 0.999, Section 9.4) and, in addition, allow a larger distance for confident, stable predictions than for uncertain ones, so that a valid photograph of many apples is not rejected for being unusual.
5. **Calibration evidence is limited.** Thresholds were derived from 100 unsupported images and were examined on the same images; the OOD analysis exists for two of the four computers. *Recommendation:* collect a fresh set of genuine photographs and about 100 new unsupported images (especially other fruit and photographs with both fruit), and evaluate the chosen thresholds on them once.
6. **Thresholds belong to one checkpoint.** They must be re-derived whenever the model changes.
7. **Image conditions.** Performance was not measured by lighting, background, fruit variety or camera. Unusual conditions may reduce accuracy.
8. **Training data differences.** Public and student-taken photographs differ in style; results on a third kind of photograph (for example professional photography) are unknown.

## 12. Using the Model

The application loads the model through `src/ml/model_service.py` and decides through `src/ml/prediction_policy.py`.

```python
from src.preprocessing.image_preprocessor import preprocess_image
from src.ml.model_service import load_model_service
from src.ml.prediction_policy import apply_policy

service = load_model_service()                       # loads weights and statistics once
image = preprocess_image(open("photo.jpg", "rb").read())   # 128 x 128 RGB, PP_v1
signals = service.predict(image)                     # probabilities, feature distance, stability
result = apply_policy(signals, service.model_version)
print(result.status.value, result.predicted_class, result.confidence)
```

**Preprocessing contract.** Inputs must be prepared exactly as `PP_v1` (EXIF orientation, RGB, bicubic resize to 128 × 128, JPEG reduced-size decode for JPEG files, pixel values 0–1). A different pipeline invalidates the model and the thresholds; the application refuses to start if the recorded preprocessing version does not match.

**Files and statistics.** The checkpoint, `final_model_selection.json` and `m4_ood_stats.npz` must come from the same run ([artifacts/README.md](../artifacts/README.md)).

## 13. References

Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

Lee, K., Lee, K., Lee, H. and Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in Neural Information Processing Systems (NeurIPS)*.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D. and Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\*)*.

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*.

Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
