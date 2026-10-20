Apple vs Banana CNN Classification Project
1. Dataset Overview
Overall Status: Pillar 1 – Dataset & Data Engineering
This dataset was prepared for the COMP6900 fruit image classification project. It contains Apple and Banana images from public and student-generated sources.
2. Dataset Statistics
•	Total images: 2,544
•	Apple: 1,272
•	Banana: 1,272
•	Student-generated: 1,272
•	Public: 1,272
3. Class and Source Balance
•	Public: Apple 636, Banana 636
•	Student-generated: Apple 636, Banana 636
•	Total: 2,544 images
The dataset contains a 50% student-generated and 50% public-source balance, with an equal number of Apple and Banana images.
4. Data Splitting
•	Training: 1,780 images — Apple 890, Banana 890
•	Validation: 382 images — Apple 190, Banana 192
•	Test: 382 images — Apple 192, Banana 190
5. Preprocessing
•	Images resized to 128 × 128 pixels.
•	Pixel values normalized to the range 0–1.
•	Batch size: 32.
6. Quality Assurance and Completed Requirements
The following Pillar 1 requirements have been completed:
•	Student-generated image collection.
•	Public dataset collection.
•	Apple and Banana class organization.
•	HEIC image conversion.
•	Image quality checking.
•	Corrupted-image checking.
•	Duplicate checking.
•	Data provenance metadata preparation.
•	Combined metadata preparation.
•	50% student-generated and 50% public-source balance.
•	50% Apple and 50% Banana class balance.
•	Training, validation and test split.
•	Image resizing to 128 × 128 pixels.
•	Pixel normalization to 0–1.
Quality Check Results
•	Student-generated dataset: No corrupted or very small images were found during the reported checks.
•	Public dataset: No corrupted or very small images were found during the reported checks.
•	Student-generated dataset: Two exact duplicate groups were identified and intentionally retained.
•	Public dataset: Zero exact duplicate groups were reported.
7. Data Provenance and Licensing
The combined dataset metadata is stored at data/metadata/combined_metadata.csv.
Public dataset source and licence information are documented in docs/data_sources_and_licensing.md.
Public dataset source: Fruit Freshness Dataset on Kaggle
The Kaggle dataset page identifies the licence as CC0 1.0 Universal (Public Domain).
Licence details: https://creativecommons.org/publicdomain/zero/1.0/
8. Limitations
The dataset contains only two fruit classes: Apple and Banana. Results from this dataset should not be assumed to generalize to other fruit classes or all real-world image conditions.
9. Intended Use
This dataset is intended for academic learning and the development and evaluation of the COMP6900 fruit image classification prototype.

