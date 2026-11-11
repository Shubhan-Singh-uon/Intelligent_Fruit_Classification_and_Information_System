<!--
File: data_sources_and_licensing.md
Project: Intelligent Fruit Classification and Information System
Description: Records the origin, licence and processing of every external
resource used by the project: image datasets, pretrained model weights,
fruit-information sources and software dependencies.
Author: Group 8
Responsible Pillar: Pillar 1 - Dataset & Data Engineering
Contributors: Wajahat Manzoor, Shubhan Singh
-->

# Data Sources and Licensing

**Project:** Intelligent Fruit Classification and Information System (COMP6900, Group 8)

This document records where every external resource used by the project came from, under what terms it may be used, and how the project processed it. It supports the [dataset card](dataset_card.md) and the [model card](model_card.md). It describes an academic prototype; it is a statement of the team's records and not legal advice.

## 1. Summary

| Resource | Origin | Terms recorded by the project |
|----------|--------|-------------------------------|
| Public fruit images (1,272 used) | Kaggle, *Fruit Freshness Dataset (Apple, Banana, Orange)* | CC0 1.0 Universal (Public Domain) |
| Student-generated fruit images (1,272 used) | Photographs taken by the project team | Team-created; no redistribution licence specified (Section 3) |
| Pretrained weights | torchvision EfficientNetB0, learned on ImageNet-1k | ImageNet terms of access: non-commercial research and education (Section 4) |
| Fruit information (nutrition, benefits, uses, facts) | 25 published sources (Section 5) | Facts summarised and attributed; each source is linked in the application |
| Software dependencies | Open-source Python packages | Open-source licences (Section 6) |

## 2. Public image dataset

| Item | Detail |
|------|--------|
| Dataset title | Fruit Freshness Dataset (Apple, Banana, Orange) |
| Platform | Kaggle |
| Uploader | user2036 |
| URL | <https://www.kaggle.com/datasets/user2036/fruit-freshness-dataset-v1> |
| Access date | 11 September 2026 |
| Classes used | Apple and Banana (the Orange class was not used) |
| Licence | CC0 1.0 Universal (Public Domain) |
| Licence URL | <https://creativecommons.org/publicdomain/zero/1.0/> |

**Processing.** The Apple and Banana images were selected for the project dataset and processed in the dataset-preparation workflow: format checks, image-quality and corruption checks, exact-duplicate checks, renaming with a source prefix and running number, and creation of provenance metadata. After class and source balancing, 636 public Apple images and 636 public Banana images were retained (1,272 in total).

**Notes on the source.**

- The Kaggle dataset page identifies the licence as CC0 1.0 Universal. The CC0 dedication permits broad reuse; attribution is not legally required, but the project records the original URL, uploader and access date to preserve provenance.
- The public images include variants that were already augmented in the source (for example flipped, rotated or translated copies of the same photograph). This is documented in the [dataset card](dataset_card.md), Section 7, because it affects how the evaluation results should be interpreted.
- Licence terms were read from the Kaggle dataset page on the access date. Platform-hosted licences can be changed by the uploader, so this record reflects the terms on that date.

## 3. Student-generated images

| Item | Detail |
|------|--------|
| Source | Photographs taken by members of the project team |
| Classes | Apple and Banana |
| Retained | 636 Apple and 636 Banana images (1,272 in total) |
| Processing | Conversion of HEIC photographs to JPEG, image-quality checks, corrupted-image checks, exact-duplicate checks and metadata preparation |
| Provenance | Per-image records in `data/metadata/combined_metadata.csv`; student images carry the identifier prefix `STU_` and public images `PUB_` |

**Rights and redistribution.** The team members who took the photographs created these images for this project. A redistribution licence for the student-generated images is not specified in the project records reviewed for this document, so until one is agreed they should be treated as available for the academic project only.

**Privacy.** The images show fruit, and no personal data is intentionally included. Before the dataset is shared outside the project, the team should confirm that no image shows an identifiable person or sensitive background detail. The application itself does not store uploaded images (see [architecture](architecture.md), Section 12).

The student-generated and public images are documented separately in the combined metadata file, which records the source type of every image.

## 4. Pretrained model weights

The classifier is an EfficientNetB0 network (Tan and Le, 2019) initialised with the torchvision weights `EfficientNet_B0_Weights.IMAGENET1K_V1`, which were learned on the ImageNet-1k dataset (Deng et al., 2009), and then trained on the project dataset. The torchvision library is distributed under a BSD-style licence. The ImageNet images themselves are made available under terms that restrict their use to **non-commercial research and educational purposes**. This project is a non-commercial academic prototype. Any commercial reuse of the trained model would require a separate review of the ImageNet terms and of the project's own data.

The trained weights produced by the project (`artifacts/M4_PC1.pt`) are derived from the pretrained weights and from the project dataset; the terms in Sections 2 to 4 therefore apply to them as well.

## 5. Fruit-information sources

The information displayed for an accepted prediction (nutrition, health benefits, uses and fun facts) is stored in the `SOURCE`, `NUTRITION`, `HEALTH_BENEFIT`, `USE` and `FUN_FACT` tables of the SQLite database (`src/database/seed_fruit.sql`). Facts were summarised by the team and attributed to a source; the application shows the source link beside each item and lists every source used in the *Sources* section of the result page. The uses of each fruit (for example "Eat fresh" or "Baking") are general descriptions and carry no source.

| ID | Source | Used for | Accessed |
|----|--------|----------|----------|
| 1 | Food Standards Australia New Zealand, Australian Food Composition Database: F000110, Apple, red skin, unpeeled, raw. <https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/search/food/F000110> | Nutrition (apple) | 2026-10-16 |
| 2 | Food Standards Australia New Zealand, Australian Food Composition Database: F000262, Banana, cavendish, peeled, raw. <https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/search/food/F000262> | Nutrition (banana) | 2026-10-16 |
| 3 | St. Vincent's Health System, "5 Reasons to Eat More Apples". <https://stvincents.org/about-us/news-press/news-detail?articleId=59557&publicid=395> | Health benefits (apple) | 2026-10-16 |
| 4 | Harvard Health Publishing, "11 Foods That Lower Cholesterol". <https://www.health.harvard.edu/heart-health/11-foods-that-lower-cholesterol-4BTQX4EV> | Health benefits (apple) | 2026-10-16 |
| 5 | Harvard T.H. Chan School of Public Health, "Apples". <https://nutritionsource.hsph.harvard.edu/food-features/apples/> | Health benefits (apple) | 2026-10-16 |
| 6 | Harvard T.H. Chan School of Public Health, "Bananas". <https://nutritionsource.hsph.harvard.edu/food-features/bananas/> | Health benefits (banana) | 2026-10-16 |
| 7 | Cleveland Clinic, "Peel It Back: Let's Look at the Health Benefits of Bananas". <https://health.clevelandclinic.org/benefits-of-bananas> | Health benefits (banana) | 2026-10-16 |
| 8 | Canadian Geographic, "Canada's Long History of Apple Growing". <https://canadiangeographic.ca/articles/canadas-long-history-of-apple-growing/> | Fun fact (apple) | 2026-10-16 |
| 9 | Wikipedia, "Rosaceae". <https://en.wikipedia.org/wiki/Rosaceae> | Fun fact (apple) | 2026-10-16 |
| 10 | Washington State University Tree Fruit, "Apple varieties". <https://treefruit.wsu.edu/web-article/apple-varieties/> | Fun fact (apple) | 2026-10-16 |
| 11 | BBC Travel, "The birthplace of the modern apple". <https://www.bbc.com/travel/article/20181120-the-birthplace-of-the-modern-apple> | Fun fact (apple) | 2026-10-16 |
| 12 | Live Science, "Why are bananas berries but strawberries aren't?". <https://www.livescience.com/57477-why-are-bananas-considered-berries.html> | Fun fact (banana) | 2026-10-16 |
| 13 | Michigan State University, "Plant science at the dinner table: bananas". <https://www.canr.msu.edu/news/plant-science-at-the-dinner-table-bananas> | Fun fact (banana) | 2026-10-16 |
| 14 | The Guardian, article on banana varieties and diversity (22 March 2024). <https://www.theguardian.com/commentisfree/2024/mar/22/1000-varieties-banana-lack-of-diversity-extinction> | Fun fact (banana) | 2026-10-16 |
| 15 | Australian Bananas, "Mind Bending Facts". <https://www.australianbananas.com.au/pages/all-about-bananas/mind-bending-facts> | Fun fact (banana) | 2026-10-16 |
| 16 | City of Ryde, "Granny Smith". <https://www.ryde.nsw.gov.au/Library/Local-and-Family-History/Historic-Ryde/Historic-Figures/Granny-Smith> | Fun fact (apple) | 2026-10-30 |
| 17 | University of Vermont Extension, "All About Apples". <https://www.uvm.edu/extension/news/all-about-apples> | Fun facts (apple, three) | 2026-10-30 |
| 18 | Royal Botanic Gardens, Kew, "Cavendish Banana". <https://www.kew.org/plants/cavendish-banana> | Fun fact (banana) | 2026-10-30 |
| 19 | San Diego Zoo, "Banana". <https://animals.sandiegozoo.org/plants/banana> | Fun fact (banana) | 2026-10-30 |
| 20 | University of Saskatchewan, "Apple Pollination". <https://gardening.usask.ca/gardening-advice/gardenline-nested-pages/food-plant-pages/fruit/apples.php> | Fun fact (apple) | 2026-10-30 |
| 21 | USDA, "Apple Nutrition and Fun Facts" (PDF). <https://www.govinfo.gov/content/pkg/GOVPUB-A98-PURL-gpo87472/pdf/GOVPUB-A98-PURL-gpo87472.pdf> | Fun fact (apple) | 2026-10-30 |
| 22 | Texas A&M AgriLife Extension, "Banana". <https://aggie-horticulture.tamu.edu/fruit-nut/fact-sheets/banana/> | Fun fact (banana) | 2026-10-30 |
| 23 | New York Botanical Garden, "Banana". <https://www.nybg.org/around-the-table/banana/> | Fun fact (banana) | 2026-10-30 |
| 24 | Michigan State University Extension, "Taste of the Tropics: Ten tropical fruits". <https://msu-prod.dotcmscloud.com/news/taste-of-the-tropics-ten-tropical-fruits> | Fun fact (banana) | 2026-10-29 |
| 25 | Harvard T.H. Chan School of Public Health, "Vitamin B6". <https://nutritionsource.hsph.harvard.edu/vitamin-b6/> | Health benefits (banana) | 2026-10-29 |

**Notes.**

- The nutrition values are per 100 g, as reported by the Australian Food Composition Database. The attribution requirements of that database should be checked at the source if the application is ever distributed beyond the course.
- Health-benefit statements are general information from the sources listed. The application states that they are not medical advice.
- The source titles above use corrected spelling; the corresponding database rows should be kept consistent with this table.

## 6. Software dependencies

| Package | Role | Licence (as declared by the project) |
|---------|------|--------------------------------------|
| PyTorch, torchvision | Model definition, training and inference | BSD-3-Clause (BSD-style) |
| Streamlit | User interface | Apache-2.0 |
| Pillow | Image reading and preprocessing | HPND (MIT-CMU style) |
| pillow-heif (optional) | Reading iPhone HEIC photographs | BSD-3-Clause; bundles libheif and codec libraries under their own licences |
| NumPy, pandas, scikit-learn | Numerical work, evaluation and covariance estimation | BSD-3-Clause |
| pytest | Automated tests | MIT |

The licences listed are those the packages declare at the time of writing; the package distributions are the authoritative source.

## 7. Attribution and compliance checklist

| Requirement | Status |
|-------------|--------|
| Source, licence and access date recorded for the public dataset | Done (Section 2) |
| Source type and provenance recorded for every image | Done (combined metadata file) |
| Source and access date recorded for every displayed fact | Done (Section 5; shown in the application) |
| ImageNet-derived weights used for non-commercial academic purposes only | Applies (Section 4) |
| Licence or consent statement for student-generated images | To be agreed before any redistribution (Section 3) |
| Confirmation that images contain no identifiable people | To be confirmed before sharing the dataset (Section 3) |

## 8. References

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. and Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
