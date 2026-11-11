<!--
File: dataset_card.md
Project: Intelligent Fruit Classification and Information System
Description: Documents dataset composition, provenance, preprocessing,
data splits, integrity checks, reproducibility, and limitations.
Author: Wajahat Manzoor
Responsible Pillar: Pillar 1 - Dataset & Data Engineering
-->
# Dataset Card: Apple and Banana Image Dataset

**Project:** Intelligent Fruit Classification and Information System (COMP6900, Group 8)

**Dataset version:** `Apple_Banana_Dataset_(Final)`

**Dataset fingerprint:** `52b395cdef84b0ff`

## 1. Summary

This dataset was prepared for the COMP6900 fruit image classification project. It contains 2,544 photographs of two classes, Apple and Banana, drawn in equal numbers from a public source and from photographs taken by the group. It was used to train and evaluate the classifier documented in the [model card](model_card.md). The format follows the idea of a datasheet for datasets (Gebru et al., 2021), adapted to the size of the project.

| Property | Value |
|----------|-------|
| Task | Binary image classification (Apple vs Banana) |
| Total images | 2,544 |
| Classes | Apple (1,272) and Banana (1,272) |
| Sources | Public (1,272) and student-generated (1,272) |
| Split | Training 1,780, validation 382, test 382 |
| Colour space | RGB photographs |
| Class labels | `0 = Apple`, `1 = Banana` (alphabetical folder order) |

## 2. Composition

### 2.1 Class and Source Balance

| Source | Apple | Banana | Total |
|--------|------:|-------:|------:|
| Public | 636 | 636 | 1,272 |
| Student-generated | 636 | 636 | 1,272 |
| **Total** | **1,272** | **1,272** | **2,544** |

The dataset is balanced 50% / 50% between public and student-generated images, and 50% / 50% between Apple and Banana. This follows the agreed rule that at least half of the images are student-generated.

### 2.2 What the Images Show

The images show whole fruit, photographed individually or in small groups against varied backgrounds, as well as cut fruit, all photographed from different angles and under different lighting conditions. The dataset carries only the fruit class (Apple or Banana). It contains no ripeness, quality or freshness labels, and it must not be used to infer them. 

_NOTE: The public source's name refers to fruit freshness, but the project uses it only for the two fruit classes._

## 3. Collection and Provenance

| Source | How it was obtained | Documentation |
|--------|--------------------|---------------|
| Public | Apple and Banana images from the Kaggle *Fruit Freshness Dataset (Apple, Banana, Orange)*, licensed CC0 1.0 Universal | [data_sources_and_licensing.md](data_sources_and_licensing.md), Section 2 |
| Student-generated | Photographs taken by members of the project team | [data_sources_and_licensing.md](data_sources_and_licensing.md), Section 3 |

Every image has a provenance record in `data/metadata/combined_metadata.csv`, including its identifier (`PUB_nnnn` for public images, `STU_nnnn` for student images), file name, class label and source type.

## 4. Preparation and Quality Assurance

### 4.1 Preparation Steps

1. Collection of student-generated and public images.
2. Organisation into Apple and Banana class folders.
3. Conversion of HEIC photographs to JPEG.
4. Image-quality, corrupted-image and very-small-image checks.
5. Exact-duplicate checks (file-content hashes).
6. Preparation of provenance metadata for each source, and of the combined metadata file.
7. Balancing to 636 images per source-and-class combination (surplus images were excluded).
8. Stratified division into training, validation and test sets (Section 5).

### 4.2 Quality-Check Results

| Check | Student-generated | Public |
|-------|-------------------|--------|
| Corrupted images | None found | None found |
| Very small images | None found | None found |
| Exact duplicate groups | Two groups identified and intentionally retained | None reported |

### 4.3 Completed Pillar 1 Requirements

| Requirement | Status |
|-------------|--------|
| Student-generated image collection | Complete |
| Public dataset collection | Complete |
| Apple and Banana class organisation | Complete |
| HEIC image conversion | Complete |
| Image quality and corrupted-image checking | Complete |
| Duplicate checking | Complete |
| Data provenance metadata (per source and combined) | Complete |
| 50% student / 50% public balance | Complete |
| 50% Apple / 50% Banana balance | Complete |
| Training, validation and test split | Complete |
| Image resizing to 128 × 128 and pixel scaling to 0–1 | Complete (applied when images are loaded; Section 6) |

## 5. Data Splits

| Split | Images | Apple | Banana | Public | Student |
|-------|-------:|------:|-------:|-------:|--------:|
| Training | 1,780 | 890 | 890 | 890 | 890 |
| Validation | 382 | 190 | 192 | 191 | 191 |
| Test | 382 | 192 | 190 | 191 | 191 |
| **Total** | **2,544** | **1,272** | **1,272** | **1,272** | **1,272** |

The division is approximately 70% / 15% / 15%. It was stratified by source and class so that each split keeps the same source and class proportions, and it used a fixed random seed (42) for reproducibility. Images are assigned to splits individually. The test set was held out from training and from hyperparameter selection: early stopping used the validation set only, and each model was evaluated on the test set once, after training.

## 6. Preprocessing

The stored image files keep their original resolution. The following steps (version `PP_v1`) are applied each time an image is loaded, for training, evaluation and inference alike:

1. Decode the image and apply its EXIF orientation, so that phone photographs are upright.
2. Convert to RGB.
3. Resize to 128 × 128 pixels with bicubic interpolation.
4. Scale pixel values to the range 0–1.

The classifier then applies the ImageNet mean and standard deviation inside the network. During training only, random horizontal flips, small rotations (±15°), random zoom (85–100% of the area), and brightness, contrast and saturation changes are applied on the fly. Hue shifts, vertical flips and noise are not used for training. No augmented image is written to disk or moved between splits. Training hyperparameters, including the batch size of 32, are documented in the [model card](model_card.md).

## 7. Known Limitations and Risks

1. **Two classes only.** The dataset contains only Apple and Banana. Results do not generalise to other fruit or to arbitrary photographs. A classifier trained on it must always choose one of the two classes, so a separate mechanism is needed to detect unsupported images (see the model card).
2. **Mostly single-fruit photographs.** The images show one fruit or a few fruit. A photograph of a large pile of apples (for example a market stall) differs from the training images; during integration testing such a photograph was rejected as unlike the known fruit images even though the classifier identified it as an apple.
3. **Pre-augmented images in the public source.** Some public images are flipped, rotated or translated variants of other public photographs, as recorded in their file names. They are part of the 1,272 public images.
4. **Split method and possible near-duplicates across splits.** Images were assigned to splits individually. Because the public source contains variants of the same photograph (item 3), a variant could in principle fall in a different split from its original. The dataset audit summarised in the training notebook reports that no augmented copies cross split boundaries, but this check is not reproduced in the dataset-preparation notebook, and has not been re-verified for this card. Until it is confirmed, the test-set results in the model card may be somewhat optimistic. **Recommended action for Pillar 1:** group each photograph with its augmented variants and confirm that every group lies within a single split.
5. **Exact duplicates.** Two exact duplicate groups in the student-generated images were intentionally retained. Only exact (byte-identical) duplicates were searched for; near-duplicates were not removed.
6. **Source differences.** Public and student-generated images differ in camera, lighting and background. The evaluation reports accuracy for each source separately, but 191 images per source in the test set is a small sample.
7. **Size.** With 382 test images, a result of 382 out of 382 correct implies only that the true accuracy is very likely above about 99.0% (exact 95% lower confidence bound), not that it is 100%.
8. **Unsupported-image (Out of Distribution (OOD)) examples.** The 100 images used to study unsupported inputs (Food 28, Other fruit 24, Ambiguous 16, Background 16, Objects 16; 50 public, 50 student) are a separate, small collection and are not part of the 2,544 images.

## 8. Intended and Out-of-Scope Use

**Intended use.** Academic learning, and the development and evaluation of the COMP6900 fruit image classification prototype.

**Out of scope.** The dataset **was** not used for ripeness, freshness, spoilage, quality or food-safety decisions; for fruit other than apples and bananas; for deployment in a commercial or safety-relevant setting; or as a benchmark for general fruit recognition.

## 9. Licensing and Provenance

The public images are licensed CC0 1.0 Universal (Public Domain); the student-generated images were taken by the project team. Source URLs, access dates, licence terms and open points are recorded in [data_sources_and_licensing.md](data_sources_and_licensing.md). Provenance for each image is in `data/metadata/combined_metadata.csv`.

## 10. Maintenance and Versioning

| Item | Value |
|------|-------|
| Dataset version label | `Apple_Banana_Dataset_(Final)` |
| Dataset fingerprint | `52b395cdef84b0ff` (recorded with every training result, so a changed dataset is detectable) |
| Preprocessing version | `PP_v1: EXIF-upright, RGB, bicubic resize 128x128, pixels 0-1` |
| Maintainer | Pillar 1 (Wajahat Manzoor) |

A change to the images, the split or the preprocessing produces a new fingerprint or preprocessing version. The model must then be retrained and re-evaluated, and the model card updated.

## 11. References

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H. and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
