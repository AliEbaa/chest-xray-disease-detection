# Chest X-Ray Disease Detection

Deep learning-based multi-label chest X-ray disease classification using DenseNet121, Transfer Learning, and a Gradio inference interface with Grad-CAM visualization.

## Project Overview

This project implements a multi-label deep learning system for detecting multiple thoracic conditions from chest X-ray images.

The system uses a pretrained **DenseNet121** model and was developed as a research and educational project to explore medical image classification, transfer learning, class imbalance handling, model evaluation, and model interpretability.

The project also includes a **Gradio-based interface** for image inference and Grad-CAM visualization.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used for medical diagnosis or clinical decision-making.

---

## Key Features

* Multi-label chest X-ray classification
* DenseNet121 with ImageNet pretrained weights
* Transfer learning and partial fine-tuning
* 15 thoracic disease/condition labels
* Class-imbalance handling using weighted BCE loss
* Class-specific prediction threshold optimization
* Macro F1-score and ROC-AUC evaluation
* Gradio-based inference interface
* Grad-CAM model interpretability visualization
* Pretrained model distributed through GitHub Releases
* Training and deployment notebooks included

---

## Disease / Condition Labels

The model predicts the following 15 labels:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax
15. No Finding

---

## Dataset

The project uses the **NIH ChestX-ray14** dataset.

The dataset contains chest X-ray images annotated with multiple thoracic conditions. The version used during development consisted of images resized to **224 × 224 pixels**.

The dataset used in the project was divided into:

| Split      |      Images |
| ---------- | ----------: |
| Training   |      78,484 |
| Validation |      16,818 |
| Test       |      16,818 |
| **Total**  | **112,120** |

The dataset itself is **not included in this repository** because of its size and distribution considerations.

---

## Model Architecture

### Backbone

**DenseNet121**

The model was initialized using ImageNet pretrained weights.

The original classifier was replaced with a fully connected layer producing 15 outputs:

```text
Linear(1024, 15)
```

Because the task is multi-label classification, each output represents an independent condition.

### Fine-Tuning

Most of the pretrained feature extractor was frozen while selected deeper DenseNet layers and the classification head were fine-tuned for the chest X-ray task.

---

## Training

### Loss Function

The model was trained using:

```text
BCEWithLogitsLoss
```

with class-specific positive weights to help address the imbalance between the different labels.

### Optimizer

```text
AdamW
```

with:

```text
Learning Rate: 3e-5
Weight Decay: 1e-5
```

### Learning-Rate Scheduler

```text
ReduceLROnPlateau
```

The training process included validation-based model selection and checkpointing.

---

## Data Preprocessing and Augmentation

Training images were processed using transformations including:

* Resize to 256 × 256
* Random crop to 224 × 224
* Random horizontal flip
* Small random rotation
* Tensor conversion
* ImageNet normalization

Validation and inference preprocessing used the required 224 × 224 input size followed by normalization.

---

## Model Performance

The best recorded validation performance was:

| Metric         |     Result |
| -------------- | ---------: |
| Macro F1-score | **0.3597** |
| Macro ROC-AUC  | **0.8214** |
| Best Epoch     |     **24** |

The reported F1-score corresponds to the best validation result used during model selection.

The test set was also prepared and used for inference, but a separate final test metric was not retained as part of the final reported experiment. Therefore, no test F1-score is claimed here.

---

## Class-Specific Thresholds

Instead of using a single probability threshold of 0.5 for every class, individual thresholds were optimized for the different labels using validation data.

The thresholds used by the final inference interface are:

```text
[0.35, 0.35, 0.45, 0.40, 0.30,
 0.40, 0.30, 0.45, 0.35, 0.40,
 0.35, 0.30, 0.25, 0.40, 0.50]
```

The threshold for **No Finding** is:

```text
0.50
```

This approach was used to account for differences in class prevalence and prediction behavior across the labels.

---

## Pretrained Model

The trained model is distributed separately through a **GitHub Release** because the model file is too large for convenient storage in the repository itself.

### Download

**Model:** `best_model.pth`

[Download the pretrained model from the latest GitHub Release](https://github.com/AliEbaa/chest-xray-disease-detection/releases/latest/download/best_model.pth)

The current release is:

**v1.0.0 — Initial Release**

After downloading the model, place it locally at:

```text
models/
└── best_model.pth
```

The model checkpoint contains the trained model parameters and supporting information used by the inference workflow.

---

## Gradio Interface

The project includes a Gradio-based interface for running inference with the trained model.

The interface provides:

* Chest X-ray image input
* Image preprocessing
* Model inference
* Multi-label predictions
* Class-specific thresholds
* Prediction visualization
* Grad-CAM visualization

The interface can be run locally after installing the required dependencies and downloading the pretrained model.

---

## Grad-CAM Interpretability

The project includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions that contributed to a model prediction.

Grad-CAM is included as a model interpretability technique to help inspect model behavior.

> **Important:** Grad-CAM visualizations should not be interpreted as clinically validated disease localization or as evidence that the highlighted region represents the true anatomical location of a disease.

An example visualization is included in:

```text
results/gradcam/
```

---

## Project Demonstration

A short demonstration of the Gradio interface is available on YouTube:

[Watch the project demonstration](https://www.youtube.com/watch?v=Y9kDzoX49wA)

---

## Repository Structure

```text
chest-xray-disease-detection/
│
├── training/
│   └── training_code.ipynb
│
├── deployment/
│   └── My_Model_Interface.ipynb
│
├── documentation/
│   └── chest-xray-disease-detection.pdf
│
├── images/
│   ├── interface.png
│   └── system_architecture.jfif
│
├── results/
│   └── Grad-Cam/
│       └── Grad-Cam.jpeg
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** `best_model.pth` is distributed through the GitHub Release and is not stored directly in the repository.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AliEbaa/chest-xray-disease-detection.git
cd chest-xray-disease-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Model

Download:

```text
best_model.pth
```

from the GitHub Release and place it in:

```text
models/
```

The resulting local structure should be:

```text
models/
└── best_model.pth
```

---

## Using the Pretrained Model

The deployment notebook contains the inference workflow.

The general workflow is:

```text
Chest X-ray Image
        ↓
Preprocessing
        ↓
DenseNet121
        ↓
15 Independent Predictions
        ↓
Class-Specific Thresholds
        ↓
Predicted Conditions
        ↓
Grad-CAM Visualization
```

---

## Training the Model

The training notebook is available under:

```text
training/
```

It contains the main workflow for:

1. Dataset preparation
2. Image preprocessing
3. Data splitting
4. Model initialization
5. Transfer learning
6. Weighted loss configuration
7. Model training
8. Validation
9. Threshold optimization
10. Model evaluation
11. Checkpoint saving

The NIH ChestX-ray14 dataset must be obtained separately.

---

## Technologies Used

### Machine Learning

* Python
* PyTorch
* Torchvision
* DenseNet121
* Transfer Learning
* Multi-label Classification
* BCEWithLogitsLoss
* AdamW
* ReduceLROnPlateau

### Computer Vision

* OpenCV
* PIL
* Grad-CAM
* Image preprocessing and augmentation

### Evaluation

* F1-score
* ROC-AUC
* Precision-Recall analysis
* Class-specific threshold optimization

### Deployment

* Gradio

### Development Environment

* Google Colab
* Jupyter Notebook

---

## Engineering and Machine Learning Challenges

The main challenges addressed during development included:

### Multi-Label Classification

A single chest X-ray can contain multiple conditions simultaneously, requiring independent predictions for each label rather than conventional single-class classification.

### Class Imbalance

The prevalence of different conditions varies considerably. Weighted binary cross-entropy was therefore used to reduce the effect of class imbalance during training.

### Threshold Selection

A single threshold does not necessarily provide suitable behavior for every class. Individual thresholds were optimized using validation data.

### Model Interpretability

Grad-CAM was incorporated to provide a visual interpretation of model predictions and help inspect model behavior.

### Deployment

The trained model was integrated into a lightweight Gradio interface to demonstrate an end-to-end inference workflow.

---

## Limitations

This project has several important limitations:

* The system is a research and educational prototype.
* It has not undergone clinical validation.
* It should not be used for diagnosis or clinical decision-making.
* Performance varies between different disease labels.
* The dataset may contain biases and annotation limitations.
* The model was trained and evaluated on a specific dataset distribution.
* External validation on independent datasets was not performed.
* Grad-CAM provides interpretability visualization but does not constitute clinically validated localization.
* The reported performance should not be interpreted as clinical diagnostic performance.

---

## Future Improvements

Potential future improvements include:

* External validation using independent datasets
* More systematic hyperparameter optimization
* Improved class-imbalance strategies
* Comparison with additional architectures
* Calibration analysis
* More comprehensive test-set evaluation
* Robustness and distribution-shift testing
* Improved model interpretability methods
* Deployment optimization for practical inference environments

---

## Documentation

The complete academic project report is available in:

```text
documentation/
```

The repository documentation is intended to provide a concise technical overview, while the full report contains additional academic and project-development details.

---

## Author

**Ali Ebaa**

Biomedical Engineering

GitHub:

https://github.com/AliEbaa

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Disclaimer

This project is provided for **educational and research purposes only**.

It is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or make clinical decisions about patients.

