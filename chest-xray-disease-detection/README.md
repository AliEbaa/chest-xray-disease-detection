# Chest X-ray Disease Detection

A deep learning-based medical imaging system for **multi-label classification of chest X-ray images** using **DenseNet121** and transfer learning.

The project provides a complete workflow covering dataset preparation, model training, evaluation, trained-model inference, and an interactive Gradio interface with Grad-CAM visualization.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical device and must not be used as a substitute for professional medical diagnosis or treatment.

---

## Project Overview

Chest X-ray imaging is widely used for the examination of thoracic conditions. The increasing availability of medical imaging datasets has created opportunities to explore deep learning methods for automated image analysis.

This project implements a **multi-label chest X-ray classification system** capable of predicting multiple conditions from a single X-ray image.

The main model is based on **DenseNet121** with transfer learning. The training pipeline also addresses class imbalance and uses class-specific decision thresholds to improve multi-label prediction performance.

The project additionally provides a **Gradio-based interface** for testing the trained model and includes **Grad-CAM** visualization to provide an interpretable view of image regions contributing to model predictions.

---

## Main Features

- Multi-label chest X-ray classification.
- DenseNet121-based deep learning model.
- Transfer learning from pretrained weights.
- 224 × 224 image input.
- Image preprocessing and data augmentation.
- Class-imbalance handling using weighted loss.
- Class-specific threshold optimization.
- Macro F1 evaluation.
- Macro ROC-AUC evaluation.
- Saved trained model for inference.
- Interactive Gradio interface.
- Grad-CAM visualization.
- Training notebook for further experimentation.
- Deployment/inference notebook for testing new images.

---

## System Workflow

```text
                    Chest X-ray Image
                           │
                           ▼
                 Image Preprocessing
                           │
                           ▼
                    DenseNet121
                 Transfer Learning
                           │
                           ▼
                  Multi-label Output
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Disease Probabilities       Grad-CAM
              │                  Visualization
              ▼                         │
      Class-specific Thresholds        │
              │                         │
              └────────────┬────────────┘
                           ▼
                   Gradio Interface
```

---

## Dataset

The project uses the **NIH ChestX-ray14** dataset.

The dataset contains more than 112,000 chest X-ray images and is widely used for research in automated chest radiograph analysis.

For this project, a resized **224 × 224** version of the images was used to reduce computational and storage requirements.

The dataset is not included in this repository because of its size.

### Dataset Classes

The implemented model uses 15 output classes:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia
15. No Finding

The additional `No Finding` class is explicitly represented in the project, resulting in 15 output classes.

### Dataset Source

**TODO:** Add the official NIH ChestX-ray14 dataset/source link here.

Example:

```markdown
[NIH ChestX-ray14 Dataset](OFFICIAL_DATASET_URL)
```

Replace `OFFICIAL_DATASET_URL` with the actual dataset URL before publishing the final version.

---

## Multi-label Classification

A chest X-ray may contain more than one abnormality at the same time.

For this reason, the project uses **multi-label classification** rather than conventional single-class classification.

For example:

```text
Atelectasis   → 0
Cardiomegaly  → 1
Effusion      → 1
Pneumonia     → 0
...
```

Each class is represented independently, allowing the model to produce a separate probability for every condition.

---

## Model Architecture

The main model used in this project is **DenseNet121**.

A pretrained DenseNet121 model is adapted to the chest X-ray classification task through transfer learning.

The final classification layer is configured for the project's 15 output classes.

### Why DenseNet121?

DenseNet121 was selected because its dense connectivity promotes feature reuse and effective gradient propagation while providing a practical architecture for transfer learning in image-classification tasks.

---

## Transfer Learning

The model starts from pretrained DenseNet121 weights rather than being trained entirely from scratch.

The project applies partial fine-tuning to adapt the pretrained network to the chest X-ray domain.

The training strategy allows previously learned visual features to be retained while adapting deeper network layers to the target medical-imaging task.

---

## Image Preprocessing

The training pipeline uses the following main transformations:

### Training

- Resize to 256 × 256.
- Random crop to 224 × 224.
- Random horizontal flip.
- Small random rotation.
- Conversion to tensor.
- ImageNet normalization.

### Validation / Inference

- Resize to 224 × 224.
- Conversion to tensor.
- ImageNet normalization.

The preprocessing pipeline is designed to provide consistent model inputs while introducing moderate augmentation during training.

---

## Class Imbalance

The NIH ChestX-ray14 dataset contains a highly imbalanced distribution of disease categories.

Some conditions have substantially more samples than others.

To reduce the effect of this imbalance, the training process uses a weighted **BCEWithLogitsLoss**.

This allows less frequent classes to have greater influence during optimization.

---

## Threshold Optimization

Because the task is multi-label classification, the model produces an independent probability for every class.

Instead of using the same decision threshold for every condition, the project searches for an appropriate threshold for each class based on F1 performance.

The resulting thresholds are stored inside the trained `best_model.pth` file together with the model state and other model information. :contentReference[oaicite:2]{index=2}

This allows the inference notebook to use the same thresholds associated with the saved model.

---

## Training

The training notebook is located at:

```text
training/
└── Training_Notebook.ipynb
```

The notebook covers the main stages of the training process:

1. Dataset loading.
2. Label processing.
3. Multi-label encoding.
4. Dataset splitting.
5. Image preprocessing.
6. Data augmentation.
7. DataLoader creation.
8. DenseNet121 initialization.
9. Transfer learning.
10. Class-weighted loss.
11. Model training.
12. Validation.
13. Threshold optimization.
14. Model selection.
15. Test-set evaluation.
16. Saving the best model.

---

## Training Configuration

The main documented training configuration is:

| Parameter | Value |
|---|---|
| Model | DenseNet121 |
| Task | Multi-label classification |
| Number of classes | 15 |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁵ |
| Loss function | BCEWithLogitsLoss |
| Model-selection metric | Macro F1 |
| Best epoch | 24 |

The project used a validation-based model-selection process, with the best model saved when the custom Macro F1 improved. :contentReference[oaicite:3]{index=3}

---

## Evaluation

The project evaluates the model using metrics suitable for multi-label classification.

### Macro F1

Macro F1 gives equal importance to each class and is useful when dealing with imbalanced multi-label datasets.

### Macro ROC-AUC

Macro ROC-AUC evaluates the model's ability to distinguish between positive and negative examples across the different classes.

### Documented Performance

| Metric | Best documented value |
|---|---:|
| Macro F1 | ≈ 0.3597 |
| Macro ROC-AUC | ≈ 0.8214 |

The reported values correspond to the documented project results and should be interpreted within the specific dataset, preprocessing, thresholding, and evaluation methodology used in this project. :contentReference[oaicite:4]{index=4}

---

## Trained Model

The main trained model is:

```text
models/
└── best_model.pth
```

The saved model contains:

- Model weights.
- Class labels.
- Class-specific thresholds.
- Best Macro F1.
- AUC value.
- Model name.
- Input size.
- Best epoch.
- Training history.

The training code explicitly saves these values together with the model state. :contentReference[oaicite:5]{index=5}

### Model Size

The model file is approximately 27 MB and is included as a standard repository file.

---

## Inference

The inference interface is located at:

```text
deployment/
└── Interface_Notebook.ipynb
```

The general inference workflow is:

```text
Input X-ray
     │
     ▼
Preprocessing
     │
     ▼
DenseNet121
     │
     ▼
Sigmoid Probabilities
     │
     ▼
Class-specific Thresholds
     │
     ▼
Predicted Conditions
```

The saved thresholds are loaded from the trained model file rather than requiring a separate thresholds file.

---

## Gradio Interface

The project includes an interactive **Gradio** interface for testing the trained model.

The interface allows the user to:

1. Upload a chest X-ray image.
2. Run the trained model.
3. View predicted conditions.
4. View prediction probabilities.
5. Generate a Grad-CAM visualization.

### Interface Preview

![Gradio Interface](images/interface.png)

> **Required file:** Save a clear screenshot of the working Gradio interface as:
>
> `images/interface.png`

---

## Grad-CAM Interpretability

The project includes **Grad-CAM (Gradient-weighted Class Activation Mapping)** as an interpretability technique.

Grad-CAM generates a visual heatmap highlighting image regions that contributed to a selected model prediction.

This provides a qualitative view of where the model's learned representation is activated for a particular prediction.

### Example

![Grad-CAM Example](results/gradcam/gradcam_example.png)

> **Required file:** Run the interface with one representative chest X-ray image, generate its Grad-CAM visualization, and save the resulting image as:
>
> `results/gradcam/gradcam_example.png`

One clear example is sufficient.

> **Important:** Grad-CAM is an interpretability method and should not be considered proof that the highlighted region represents the true pathological finding.

---

## System Architecture

The overall architecture of the project is illustrated below.

![System Architecture](images/system_architecture.png)

> **Required file:** Extract the final system architecture figure from the graduation project documentation and save it as:
>
> `images/system_architecture.png`

The figure should be the final version of the architecture diagram used in the project.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AliEbaa/chest-xray-disease-detection.git
cd chest-xray-disease-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Training Pipeline

### 1. Obtain the Dataset

Download the NIH ChestX-ray14 dataset separately.

### 2. Configure the Dataset Path

Open:

```text
training/Training_Notebook.ipynb
```

Update the dataset paths according to your local environment.

The original development environment used Google Colab and Google Drive, so paths such as `/content/drive/...` may need to be changed when running the notebook locally.

### 3. Run the Notebook

Execute the notebook cells sequentially to:

- Prepare the dataset.
- Train the model.
- Evaluate performance.
- Select the best model.
- Save the trained model.

> **Computational note:** Training a DenseNet121 model on the full dataset can require substantial computational resources and may take considerably longer on CPU-only systems.

---

## Running the Inference Interface

To test the pretrained model:

1. Install the required packages.
2. Make sure `models/best_model.pth` is available.
3. Open:

```text
deployment/Interface_Notebook.ipynb
```

4. Update any local file paths if required.
5. Run the notebook.
6. Launch the Gradio interface.
7. Upload a chest X-ray image.
8. Review the model predictions.
9. Generate a Grad-CAM visualization if desired.

The inference process does not require retraining the model.

---

## Repository Structure

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
│       └── gradcam_example.png
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

---

## Demonstration

A short demonstration of the trained model and Gradio interface is available on YouTube.

> **TODO:** Replace the placeholder URL below with the actual YouTube video link after uploading the demonstration.

[Watch the project demonstration]((https://www.youtube.com/watch?v=Y9kDzoX49wA))

---

## Limitations

### Dataset Resolution

The project uses resized 224 × 224 images rather than the original high-resolution images. This reduces computational requirements but may result in the loss of fine-grained visual information.

### Class Imbalance

The dataset contains significant differences in the number of samples available for different conditions.

### Dataset Labels

Labels derived from large-scale chest X-ray datasets may contain uncertainty and should not necessarily be considered equivalent to independently verified clinical ground truth.

### Generalization

The model was developed and evaluated using the selected dataset and experimental setup. Performance on external datasets or different clinical environments may differ.

### Clinical Validation

The system has not undergone clinical validation and is not intended for direct clinical decision-making.

---

## Future Improvements

Potential future improvements include:

- Training with higher-resolution images.
- Exploring alternative CNN and transformer architectures.
- More advanced class-imbalance strategies.
- Improved probability calibration.
- External validation using independent datasets.
- More extensive interpretability analysis.
- Additional explainability techniques.
- Development of a dedicated web or mobile application.
- Integration with larger clinical workflows.
- More extensive model robustness testing.

---

## Educational Context

This project was developed as a graduation project in **Biomedical Engineering**.

It combines concepts from:

- Biomedical Engineering
- Medical Imaging
- Deep Learning
- Computer Vision
- Machine Learning
- Transfer Learning
- Multi-label Classification
- Explainable AI
- Model Evaluation

The project was developed using open-source tools and was designed to demonstrate an end-to-end medical AI workflow.

---

## References

### Dataset

Wang, X. et al.

**ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.**

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

### DenseNet

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q.

**Densely Connected Convolutional Networks.**

Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

### Grad-CAM

Selvaraju, R. R. et al.

**Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.**

International Conference on Computer Vision (ICCV), 2017.

For additional references and detailed methodological discussion, see the project report included in the `documentation/` directory.

---

## Project Documentation

The complete graduation project report is available at:

```text
documentation/Project_Report.pdf
```

The report contains the detailed methodology, dataset analysis, training procedure, experimental results, and technical discussion.

---

## Author

**Ali Ebaa Hasan**

Biomedical Engineering

University of Latakia

GitHub:

https://github.com/AliEbaa

---

## License

This project is released under the **MIT License**.

See the `LICENSE` file for the complete license text.

---

## Disclaimer

This project is provided for educational and research purposes only.

The predictions generated by the model should not be interpreted as medical diagnoses.

The system has not been clinically validated and should not be used to make decisions concerning patient diagnosis, treatment, or patient care.
