<!--
File: README.md
Project: Intelligent Fruit Classification and Information System
Description: Provides the project overview, repository structure,
environment setup, database initialisation, model requirements,
application launch instructions, and usage guidance.
Author: Group 8
Responsible Area: Shared/Core
-->
# Intelligent Fruit Classification and Information System

A Python application that classifies a picture as an **apple** or a **banana** with a convolutional neural network (CNN), judges whether the prediction can be trusted, and, for trustworthy predictions, shows information about the fruit (nutrition, health benefits, uses, fun facts and its sources) from an SQLite database. Every attempt is recorded in a prediction history.

Developed for **COMP6900: Computing Project (Project 4)** at the University of Newcastle by Group 8 in Semester 2 - 2026.

## Contents

1. [Features](#1-features)
2. [Scope and Limitations](#2-scope-and-limitations)
3. [How it works](#3-how-it-works)
4. [Requirements](#4-requirements)
5. [Installation](#5-installation)
6. [Model files](#6-model-files)
7. [Initialising the database](#7-initialising-the-database)
8. [Running the application](#8-running-the-application)
9. [Using the application](#9-using-the-application)
10. [Running the tests](#10-running-the-tests)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [Repository Structure](#13-repository-structure)
14. [Documentation](#14-documentation)
15. [Team](#15-team)
16. [Academic context, licences and acknowledgements](#16-academic-context-licences-and-acknowledgements)

## 1. Features

- **Classification** of an uploaded photograph as Apple or Banana, with both class probabilities and a confidence score. Reads JPG, PNG, WEBP, BMP, GIF, TIFF and (with an optional package) HEIC files, recognised by their content and not by their extension.
- **Reliability assessment.** Each prediction is assigned one of four outcomes (*Accepted*, *Needs review*, *Rejected*, *Invalid input*), using the model's confidence, the distance of the image from the known fruit images, and the stability of the prediction. A photograph of a landscape is rejected and not forced into one of the two classes.
- **Fruit information** for accepted predictions: nutrition per 100 g, health benefits, common uses, fun facts and the sources behind each item.
- **Prediction history** with summary counts, status, confidence and time, read from the database.
- **Separation of outcomes from faults.** A database or model problem is reported as a system problem and never as a classification result.

## 2. Scope and Limitations

The system tells apples and bananas apart. It does **not** assess ripeness, quality, freshness or food safety, and it does not recognise any other fruit. It is an academic prototype and must not be used for health, dietary or safety decisions.

On the held-out test set of 382 images the selected model classified all 382 correctly (the exact 95% lower bound for accuracy is about 99.0%). The result should be read with care: the test set is small, and the model card lists further caveats, including a possible overlap between near-identical images in the training and test sets and the trade-offs of the rule that rejects unsupported images. See the [model card](docs/model_card.md).

## 3. How it works

1. The user uploads an image on the **Classify Fruit** page.
2. The image is validated and prepared exactly as the training images were: EXIF orientation, RGB, bicubic resize to 128 × 128, pixel values 0–1.
3. A fine-tuned **EfficientNetB0** network (PyTorch) returns the probabilities of Apple and Banana, together with two reliability signals.
4. A decision policy assigns the outcome.
5. For an accepted prediction, the fruit information is read from **SQLite** and displayed.
6. The attempt is saved to the prediction history.

The model is trained offline in notebooks; the application only loads the finished model and never retrains it. A fuller description, with diagrams, is in the [architecture document](docs/architecture.md).

## 4. Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.9 or later (developed on 3.9 under Windows 11; also tested on 3.12 under Linux) |
| Hardware | An ordinary CPU is sufficient; no GPU is needed. Classifying one image takes well under a second |
| Disk space | About 1–2 GB for the Python packages (mostly PyTorch) |
| Network | Needed only to install packages; the application makes no network request to classify an image |
| Git | To clone the repository |

The dataset images are **not** needed to run the application. They are needed only to repeat the model training in the notebooks.

## 5. Installation
Run all commands from the project root folder. A virtual environment keeps the project's packages separate from the rest of the computer.

### 5.1 Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```
### 5.2 Install Dependencies
Once the virtual environment is activated ( (.venv) in visible in your terminal), install the required packages:
```
pip install -r requirements.txt
```

`pillow-heif` in `requirements.txt` is optional: it lets the application read iPhone HEIC photographs. If it cannot be installed on your system, remove that line; everything else works without it.

## 6. Model files

The application needs three files in the `artifacts/` folder: the trained weights (`M4_PC1.pt`), the deployment record (`final_model_selection.json`) and the feature statistics (`m4_ood_stats.npz`). If they are present in your copy of the repository, no further action is needed. Otherwise, follow [`artifacts/README.md`](artifacts/README.md), which explains where each file comes from and how to generate the statistics file.

At start-up the application checks that the three files are present and consistent with one another. If they are not, it shows a clear error and stops, instead of producing unreliable predictions.

## 7. Initialising the database

Create the database once, before the first run:

```powershell
python src/scripts/init_db.py
```

The script builds `src/database/fruit_classifier.db` from `src/database/schema.sql` and `src/database/seed_fruit.sql`, then lists the tables it created. Expected output:

```text
Database created successfully.
Tables in database:
FRUIT
FUN_FACT
HEALTH_BENEFIT
NUTRITION
PREDICTION_HISTORY
SOURCE
USE
```

| Command | Effect |
|---------|--------|
| `python src/scripts/init_db.py` | Creates the database. If it already exists, nothing is changed and the script explains how to reset it |
| `python src/scripts/init_db.py --reset` | Deletes the database and rebuilds it. **This erases the prediction history** |

The database file is generated locally and is not committed to version control: every installation creates its own. Run `--reset` after pulling a change to `schema.sql` (for example a new column), because the script does not alter an existing database.

## 8. Running the application

```powershell
streamlit run app.py
```

Streamlit starts a local server and opens the application in the browser, normally at <http://localhost:8501>. The first start takes a few seconds while the model loads. Press `Ctrl+C` in the terminal to stop the server. To use another port: `streamlit run app.py --server.port 8502`.

## 9. Using the application

The application has three pages, shown in the sidebar.

| Page | Purpose |
|------|---------|
| **Home** | Introduction, how it works and the supported fruit |
| **Classify Fruit** | Upload an image, preview it, press **Classify Image**, and read the result |
| **Prediction History** | Counts of attempts by outcome, and a table of past attempts with time, result, confidence, status and reason |

### 9.1 Classifying an image

1. Open **Classify Fruit** and drag an image into the upload area, or choose a file. Supported formats: JPG, PNG, WEBP, BMP, GIF, TIFF and HEIC, up to 5 MB.
2. Check the preview and press **Classify Image**.
3. Read the result. Use **Try Another Image** to start again.

### 9.2 The four outcomes

| Outcome | What it means | What you see |
|---------|---------------|--------------|
| **Accepted** | The image resembles the training images and the model is confident and consistent | The fruit, both probabilities, the confidence, and the fruit information (nutrition, health benefits, uses, fun facts, sources) |
| **Needs review** | The image may be valid, but the model is not confident enough, or its answer changes when the image is slightly altered | A tentative match with probabilities and a warning; no fruit information. Try a clearer image or another angle |
| **Rejected** | The image looks too different from the apples and bananas the system was built for (for example another fruit, an object or a landscape) | A message that no fruit was identified; no fruit information. The model's raw numbers are in a collapsed *Technical details* section |
| **Invalid input** | The file cannot be read as an image, is empty, is too large or is in an unsupported format | A specific error message and a retry option |

If the fruit information or the history cannot be loaded or saved, the result is still shown, with a warning explaining what could not be done.

## 10. Running the tests

```powershell
pytest -v
```

The suite does not need the model files or the dataset, because the model tests use a small randomly initialised network with the same architecture. It covers image handling, the model service, the decision policy, the prediction service (with fake components and against the real repositories and SQL files) and the interface pages. 

Use `pytest -v` for all tests to run, or `pytest tests/test_prediction_policy.py` to run a single test file. **All tests are expected to pass.**

**Pillar 4 tests (Hasnain Younas).** `tests/test_hasnain.py` contains 14 tests for `src/services/prediction_service.py` and the Home and Prediction History pages. They use a fake model and build a new SQLite database from `schema.sql` and `seed_fruit.sql` for each test, so they finish in about a second and never change the real database. They check that fruit information appears only for Accepted results, that every outcome is saved to the history (a model crash saves nothing), that a result is still shown when the database is unavailable, that the History summary counts each outcome correctly, and that the pages work with a missing or older database. Run them on their own with:

```powershell
pytest tests/test_hasnain.py -v
```

## 11. Configuration

The settings are in `src/fruit_classifier/config.py`. Changes affect every part of the application and should be agreed by the team.

| Setting | Purpose |
|---------|---------|
| `CONFIDENCE_THRESHOLD` | Confidence below which a prediction is sent to review |
| `DISTANCE_REJECT_THRESHOLD` | Distance from the known fruit images beyond which an image is rejected |
| `ALLOWED_IMAGE_FORMATS`, `UPLOAD_EXTENSIONS`, `MAX_UPLOAD_BYTES` | Which files are accepted and the size limit |
| `SAVE_TO_HISTORY_STATUSES` | Which outcomes are stored in the history |

The two thresholds belong to one trained model. How they were chosen, and what changing them does, is explained in the [model card](docs/model_card.md), Section 9. To re-examine them, for example after changing the model, run the calibration script described in [`artifacts/README.md`](artifacts/README.md).

The header picture of the Home page is `assets/home_header.png` (see `assets/README.md`). The Streamlit theme is in `.streamlit/config.toml`, which must stay in the project root.

## 12. Troubleshooting

| Problem | Likely cause and remedy |
|---------|-------------------------|
| `ModuleNotFoundError: No module named 'src'` | A command was run from the wrong folder. Run all commands from the project root |
| "The classification model could not be loaded" | A model file is missing or inconsistent. See [`artifacts/README.md`](artifacts/README.md), Section 7 |
| The History page says the history could not be read | The database has not been created. Run `python src/scripts/init_db.py` |
| The history has no *Date & Time* column | The database was created before the timestamp column was added. Run `python src/scripts/init_db.py --reset` (this erases the history) |
| Buttons are red and the theme is missing | `.streamlit/config.toml` is not in the project root, or Streamlit was started from another folder |
| A HEIC photograph is reported as unreadable | Install the optional package: `pip install pillow-heif` |
| "Port 8501 is not available" | Another instance is running. Stop it, or use `--server.port 8502` |
| Odd behaviour after editing source files while the app runs | Stop the server (`Ctrl+C`) and start it again |
| Installation downloads several gigabytes on Linux | Install the CPU-only PyTorch first (Section 5.2) |
| PowerShell will not activate the environment | Allow local scripts (Section 5.1) |

## 13. Repository structure

```text
Intelligent_Fruit_Classification_and_Information_System/
├── app.py                          Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
├── artifacts/                      Model files (see artifacts/README.md)
├── data/                           Dataset metadata (images are not needed to run the app)
├── docs/                           architecture, model card, dataset card, licensing
├── notebooks/                      Dataset preparation, training and collation notebooks
├── src/
|   ├── assets/                     Home page header picture
│   ├── fruit_classifier/           config.py, schemas.py, exceptions.py
│   ├── preprocessing/              image_preprocessor.py
│   ├── ml/                         model_service.py, prediction_policy.py
│   ├── services/                   prediction_service.py
│   ├── repositories/               database.py, fruit_repository.py, prediction_repository.py
│   ├── database/                   schema.sql, seed_fruit.sql
│   ├── scripts/                    init_db.py, export_ood_stats.py
│   └── ui/                         home_page.py, classify_page.py, history_page.py, styles.py, messages.py
└── tests/                          pytest suite
```

## 14. Documentation

| Document | Contents |
|----------|----------|
| [`docs/architecture.md`](docs/architecture.md) | Components, workflow, decision policy, database, error handling and design decisions |
| [`docs/model_card.md`](docs/model_card.md) | The model, its evaluation, how it was selected, its thresholds and its limits |
| [`docs/dataset_card.md`](docs/dataset_card.md) | The dataset: composition, preparation, splits and limitations |
| [`docs/data_sources_and_licensing.md`](docs/data_sources_and_licensing.md) | Origin and licence of the datasets, pretrained weights, fruit-information sources and software |
| [`artifacts/README.md`](artifacts/README.md) | The model files and how to generate, verify and replace them |

## 15. Team

| Pillar | Area | Member |
|--------|------|--------|
| 1 | Dataset and data engineering | Wajahat Manzoor |
| 2 | Machine learning and evaluation | Shubhan Singh |
| 3 | Database and information system | Abeeha Mariam |
| 4 | Application and system integration | Hasnain Younas |

All four members contributed to and verified the fruit-information content.

## 16. Academic context, licences and acknowledgements

This project was developed for the course COMP6900 as an academic prototype. It is not intended for production, commercial or safety-relevant use, and its fruit information is general knowledge and not medical or dietary advice.

No licence has been specified for the project's source code. The datasets, pretrained weights, information sources and software dependencies are subject to their own terms, which are recorded in [`docs/data_sources_and_licensing.md`](docs/data_sources_and_licensing.md). In particular, the public images come from the Kaggle *Fruit Freshness Dataset* (CC0 1.0), and the network was initialised with ImageNet-pretrained EfficientNetB0 weights from torchvision, which are provided for non-commercial research and education.

## **Key references**

Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.

Lee, K., Lee, K., Lee, H. and Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.

Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.

Gebru, T., et al. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12).

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*.
