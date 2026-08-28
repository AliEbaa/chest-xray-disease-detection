# Chest X-Ray Disease Detection

A deep learning-based multi-label classification system for detecting multiple thoracic diseases from chest X-ray images using **DenseNet121**. The project includes a complete training pipeline, a pretrained model, a Gradio-based inference interface, and Grad-CAM visualization for model interpretability.

> **Note:** This project was developed as an academic and educational machine learning project. It is not intended for clinical diagnosis or direct medical decision-making.

---

# Project Overview

Chest X-ray imaging is one of the most widely used medical imaging modalities for examining thoracic conditions. A single X-ray image may contain evidence of more than one abnormality, making **multi-label classification** an important task in medical image analysis.

This project implements a deep learning system capable of analyzing chest X-ray images and predicting the presence of multiple thoracic conditions simultaneously.

The system uses **DenseNet121 with transfer learning** and is trained on the **NIH ChestX-ray14** dataset. A dedicated Gradio interface is provided for model inference, while Grad-CAM is used to provide visual explanations of the model's predictions.

The project demonstrates an end-to-end Medical AI workflow covering:

* Medical image preprocessing
* Multi-label classification
* Transfer learning
* Model training and evaluation
* Threshold optimization
* Model inference
* Explainable AI using Grad-CAM
* Interactive deployment using Gradio

---

# Key Features

* Multi-label chest X-ray disease classification.
* DenseNet121-based deep learning architecture.
* Transfer learning using pretrained ImageNet weights.
* Training with class-imbalance handling.
* BCEWithLogitsLoss with positive-class weighting.
* Individual prediction thresholds for different labels.
* ROC-AUC and F1-based model evaluation.
* Grad-CAM visualization for model interpretability.
* Interactive Gradio inference interface.
* Pretrained model available through GitHub Releases.
* Complete training and inference code.
* Reproducible Python environment through `requirements.txt`.

---

# Diseases / Labels

The model predicts the following 15 labels:

| #  | Label              |
| -- | ------------------ |
| 1  | Atelectasis        |
| 2  | Cardiomegaly       |
| 3  | Consolidation      |
| 4  | Edema              |
| 5  | Effusion           |
| 6  | Emphysema          |
| 7  | Fibrosis           |
| 8  | Hernia             |
| 9  | Infiltration       |
| 10 | Mass               |
| 11 | Nodule             |
| 12 | Pleural Thickening |
| 13 | Pneumonia          |
| 14 | Pneumothorax       |
| 15 | No Finding         |

---

# Dataset

The project is based on the **NIH ChestX-ray14** dataset.

The dataset contains chest X-ray images annotated with multiple thoracic disease labels and is commonly used for research in medical image classification.

The original dataset is **not included in this repository** because of its size and dataset distribution considerations.

Users wishing to reproduce the training process should obtain the dataset from its official/research distribution source and configure the dataset paths according to the training code.

---

# Model Architecture

The project uses **DenseNet121** with transfer learning.

The original ImageNet classification head is replaced with a fully connected layer producing 15 outputs corresponding to the target labels.

The model performs multi-label classification using independent output probabilities for each disease.

### Architecture Overview

```text
Chest X-Ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
DenseNet121
        │
        ▼
Feature Extraction
        │
        ▼
Fully Connected Layer
        │
        ▼
15 Independent Outputs
        │
        ▼
Disease Predictions
```

---

# Training

The training pipeline includes:

* Image resizing and normalization.
* Data augmentation for training images.
* Transfer learning using DenseNet121.
* Partial freezing of the pretrained network.
* Class-imbalance handling using positive-class weighting.
* BCEWithLogitsLoss.
* AdamW optimization.
* Learning-rate scheduling.
* Validation monitoring.
* F1-score and ROC-AUC evaluation.
* Early stopping / best-model selection.

The main training code is provided in the repository under the training-related files.

---

# Model Performance

The model was evaluated using multi-label classification metrics, with particular emphasis on **F1-score** and **ROC-AUC**.

The final trained model and evaluation information are provided as part of the project files and documentation.

Because medical datasets are highly imbalanced, accuracy alone is not considered an adequate primary evaluation metric for this task.

---

# Pretrained Model

A pretrained version of the trained model is provided through the project's GitHub Release.

### Download

**[Download the pretrained `best_model.pth`](https://github.com/AliEbaa/chest-xray-disease-detection/releases/latest/download/best_model.pth)**

After downloading the model, place it in the following location:

```text
models/
└── best_model.pth
```

The pretrained model can then be loaded by the inference/deployment code.

### Release

The current model release is:

**v1.0.0 – Initial Release**

---

# Gradio Interface

The project includes an interactive **Gradio** interface for testing the trained model.

The interface allows users to:

1. Upload a chest X-ray image.
2. Preprocess the image.
3. Run inference using the trained DenseNet121 model.
4. Display predicted conditions.
5. Apply the optimized prediction thresholds.
6. Generate Grad-CAM visualizations for model interpretability.

The interface-related code is included in the repository.

---

# Grad-CAM Interpretability

Grad-CAM (**Gradient-weighted Class Activation Mapping**) is used to provide a visual explanation of model predictions.

The technique generates a heatmap highlighting image regions that contributed to a particular prediction.

This provides an additional interpretability layer and allows the user to inspect which areas of the chest X-ray influenced the model's output.

> Grad-CAM visualizations are intended for model interpretability and debugging. They should not be interpreted as clinically validated localization of disease.

Example Grad-CAM outputs and other project images are available in the repository's image/result directories.

---

# Project Demonstration

A short demonstration of the Gradio interface and model inference is available on YouTube:

**[Watch the Project Demonstration](https://www.youtube.com/watch?v=Y9kDzoX49wA)**

---

# Repository Structure

```text
chest-xray-disease-detection/
│
├── training/
│   └── Training_Notebook.ipynb
│
├── deployment/
│   └── Interface_Notebook.ipynb
│
├── models/
│   └── best_model.pth
│
├── results/
│   └── gradcam/
│
├── images/
│   ├── system_architecture.png
│   └── interface.png
│
├── documentation/
│   └── Project_Report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** The pretrained `best_model.pth` file is distributed through GitHub Releases rather than stored directly in the repository because of GitHub's file-upload limitations.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AliEbaa/chest-xray-disease-detection.git
cd chest-xray-disease-detection
```

## 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Using the Pretrained Model

Download:

```text
best_model.pth
```

from the project's GitHub Release and place it inside:

```text
models/
```

The resulting structure should be:

```text
models/
└── best_model.pth
```

Then run the provided inference/deployment code.

---

# Training the Model

Users who wish to reproduce the training process can use the provided training code.

The general workflow is:

```text
NIH ChestX-ray14 Dataset
          │
          ▼
Data Preparation
          │
          ▼
Image Preprocessing
          │
          ▼
Data Augmentation
          │
          ▼
DenseNet121 Transfer Learning
          │
          ▼
Model Training
          │
          ▼
Validation
          │
          ▼
Threshold Optimization
          │
          ▼
Best Model
```

Before training, configure the dataset paths according to the environment being used.

The pretrained model is provided separately for users who only want to perform inference without retraining the network.

---

# Inference Workflow

The inference process follows:

1. Load the trained DenseNet121 model.
2. Load a chest X-ray image.
3. Apply the required preprocessing.
4. Perform model inference.
5. Apply the predefined prediction thresholds.
6. Display predicted labels.
7. Generate Grad-CAM visualization when requested.

---

# Technologies Used

| Technology   | Purpose                                   |
| ------------ | ----------------------------------------- |
| Python       | Main programming language                 |
| PyTorch      | Deep learning framework                   |
| Torchvision  | DenseNet121 and image processing          |
| NumPy        | Numerical processing                      |
| Pandas       | Dataset and tabular data processing       |
| OpenCV       | Image processing                          |
| PIL          | Image loading and preprocessing           |
| Matplotlib   | Visualization                             |
| Gradio       | Interactive inference interface           |
| Google Colab | Development and model training            |
| GitHub       | Version control and project documentation |

---

# Engineering and Machine Learning Challenges

Several challenges were addressed during the development of the project, including:

* Handling a highly imbalanced medical imaging dataset.
* Designing a multi-label classification pipeline.
* Selecting appropriate evaluation metrics.
* Optimizing individual prediction thresholds.
* Adapting DenseNet121 for multi-label classification.
* Managing computational limitations during model training.
* Developing an interactive inference interface.
* Adding model interpretability using Grad-CAM.

---

# Limitations

This project has several important limitations:

* The model is trained for research and educational purposes.
* It has not undergone clinical validation.
* Predictions should not be used as a substitute for professional medical diagnosis.
* Dataset imbalance may affect performance across different diseases.
* Model performance may vary on images from different institutions, devices, or patient populations.
* The reported performance should not be interpreted as evidence of clinical effectiveness.

---

# Future Improvements

Potential future developments include:

* Training with larger and more diverse medical datasets.
* External validation on independent datasets.
* Improved calibration of model probabilities.
* More advanced explainability techniques.
* Model optimization for deployment on edge devices.
* Integration with medical imaging standards such as DICOM.
* Development of a more complete medical AI deployment pipeline.
* Clinical validation and prospective evaluation.

---

# Documentation

Additional project documentation, including the complete graduation project report, is available in the repository.

The documentation provides further information about:

* Dataset preparation
* Methodology
* Model architecture
* Training process
* Evaluation
* Interface implementation
* Experimental results
* System limitations

---

# Author

**Ali Ebaa Hasan**

Biomedical Engineering

University of Latakia, Syria

GitHub:

**[AliEbaa](https://github.com/AliEbaa)**

---

# License

This project is released under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

# Disclaimer
