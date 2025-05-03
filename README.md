# Green Cloud AI

# 🌱 Green Cloud AI: Data Pipeline with DVC and MLflow

This project demonstrates how to build an end-to-end reproducible Machine Learning pipeline tailored specifically for **Green Cloud AI**, using **DVC (Data Version Control)** for versioning data and models, and **MLflow** for tracking experiments. The pipeline utilizes a **Random Forest Classifier** trained on a CPU performance metrics dataset sourced from Kaggle, focusing on resource optimization and sustainable cloud computing practices.
Access the complete project on DagsHub:

🔗 [Green Cloud AI Pipeline Repository](https://dagshub.com/LeerishArvind/Green_Cloud_AI)
---

## 🎯 Project Objectives

- **Reproducibility:** Achieve reliable and consistent ML outcomes.
- **Optimization:** Use MLflow to track and optimize model performance.
- **Sustainability:** Aid resource optimization in cloud environments.

---

## ⚙️ Key Features

### ✅ Data Version Control (DVC)
- Version control for data, pipeline stages, and ML models.
- Automated pipeline execution when changes occur in dependencies.
- Integration with remote data storage (DagsHub, S3).

### ✅ Experiment Tracking with MLflow
- Logs model parameters and evaluation metrics.
- Provides comparisons across experiments for informed model tuning.

---

## 🚧 Pipeline Stages

### 🛠️ Preprocessing

**Script:** `src/preprocess.py`

- Reads CPU metrics data (`data/raw/cpu_metrics.csv`).
- Cleans and processes data, saving the output to `data/processed/cpu_processed.csv`.

### 📊 Training

**Script:** `src/train.py`

- Trains Random Forest Classifier on processed data.
- Saves model artifact to `models/random_forest_cpu.pkl`.
- Logs hyperparameters and models to MLflow.

### 🔍 Evaluation

**Script:** `src/evaluate.py`

- Evaluates model accuracy and performance.
- Logs evaluation metrics in MLflow.

---

## ⚡ Quickstart Commands

### Add Pipeline Stages with DVC:

```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/cpu_metrics.csv \
    -o data/processed/cpu_processed.csv \
    python src/preprocess.py


dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/processed/cpu_processed.csv \
    -o models/random_forest_cpu.pkl \
    python src/train.py


dvc stage add -n evaluate \
    -d src/evaluate.py -d models/random_forest_cpu.pkl -d data/processed/cpu_processed.csv \
    python src/evaluate.py
```

---

## 🚀 Project Goals & Benefits

- **Collaboration:** Structured environment for team workflows.
- **Traceability:** Easy management of changes and experiments.
- **Efficiency:** Rapid experiment iterations and clear metric tracking.

---

## 📌 Use Cases

- **Cloud Optimization:** Manage and optimize CPU resource usage.
- **Performance Analytics:** Predictive analysis for cloud infrastructures.
- **Research Applications:** Facilitate reproducible sustainability research.

---

## 🧰 Tech Stack

- **Python**: Pipeline scripting language.
- **DVC**: Version control for ML workflows.
- **MLflow**: Experiment tracking and model management.
- **Scikit-learn**: Implementation of Random Forest Classifier.

---

## 🌐 Project Repository

Access the complete project on DagsHub:

🔗 [Green Cloud AI Pipeline Repository](https://dagshub.com/LeerishArvind/Green_Cloud_AI)

---

This structured pipeline provides a solid foundation for efficient and sustainable cloud resource management through Machine Learning.



## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://GreenCloudAI.streamlit.app/)


