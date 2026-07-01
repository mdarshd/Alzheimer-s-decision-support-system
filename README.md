<div align="center">

# 🧠 Alzheimer's Clinical Decision Support System

### AI-Powered Multi-Modal Healthcare Platform for Early Alzheimer's Disease Screening

<p align="center">

An intelligent Clinical Decision Support System (CDSS) that combines
<b>Machine Learning</b>,
<b>Deep Learning</b>,
<b>Computer Vision</b>, and
<b>Generative AI</b>
to assist healthcare professionals in Alzheimer's Disease risk assessment and MRI-based disease staging.

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">

<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch">

<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn">

<img src="https://img.shields.io/badge/Groq-Llama%203.1-blueviolet?style=for-the-badge">

<img src="https://img.shields.io/badge/Tkinter-Desktop%20Application-success?style=for-the-badge">

<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">

</p>

</div>

---

# 📖 Overview

Alzheimer's Disease is one of the world's fastest-growing neurodegenerative disorders, where early diagnosis plays a critical role in slowing disease progression and improving patient care. Traditional diagnosis requires cognitive assessments, clinical evaluation, and MRI interpretation by specialists, making the process both resource-intensive and time-consuming.

This project introduces an **AI-powered Clinical Decision Support System (CDSS)** that integrates structured clinical data, MRI image analysis, and Explainable Artificial Intelligence into a unified diagnostic workflow.

Unlike standalone prediction models, this system combines multiple AI technologies to provide a comprehensive clinical assessment with transparent, human-readable explanations.

The platform is designed to demonstrate how modern AI can support healthcare professionals while maintaining interpretability and modular software architecture.

---

# 🎯 Objectives

- Detect Alzheimer's disease risk using clinical patient information.
- Classify MRI brain scans into disease severity stages.
- Generate explainable AI-assisted diagnostic summaries.
- Provide personalized healthcare recommendations.
- Demonstrate production-oriented AI software engineering.

---

# ✨ Key Features

## Clinical Risk Prediction

✔ Random Forest classifier trained on structured clinical datasets

✔ Cognitive assessment using MMSE score

✔ Demographic and lifestyle analysis

✔ Medical history evaluation

✔ Risk estimation before MRI analysis

---

## MRI Disease Classification

Fine-tuned **ResNet18 Convolutional Neural Network**

Supports automatic classification into:

- Normal
- Mild Dementia
- Moderate Dementia
- Severe Dementia

Includes

- Image preprocessing
- Normalization
- Deep learning inference
- Confidence prediction

---

## Explainable Artificial Intelligence

Black-box AI systems reduce trust in healthcare.

To improve interpretability, this project integrates **Groq Llama 3.1** which converts model outputs into clinically meaningful explanations.

Generated reports include

- Diagnostic reasoning

- Clinical interpretation

- MRI explanation

- Plain-language summary

- Important medical disclaimer

---

## Recommendation Engine

The recommendation system generates personalized guidance based on disease stage.

Recommendations include

- Lifestyle improvements

- Diet suggestions

- Memory exercises

- Preventive care

- Clinical follow-up

---

## Desktop Application

Interactive desktop interface developed using **Tkinter**.

Supports

- Clinical questionnaire

- MRI upload

- Real-time prediction

- AI explanation

- Recommendation dashboard

---

# 🏗 System Architecture

```

Clinical Data
│
▼
Random Forest Prediction
│
├───────────────► Low Risk
│
▼
High Risk
│
▼
MRI Upload
│
▼
ResNet18 CNN
│
▼
Disease Stage
│
▼
Groq LLM
│
▼
Clinical Explanation
│
▼
Recommendation Engine

```

---

# ⚙ Technology Stack

| Category | Technologies |
|------------|-------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision |
| Data Processing | Pandas, NumPy |
| Explainable AI | Groq API, Llama 3.1 |
| Desktop Development | Tkinter |
| Environment | python-dotenv |
| Version Control | Git, GitHub |

---

# 📂 Repository Structure

```text
.
├── app.py
│
├── prediction/
│   ├── csv_prediction.py
│   ├── clinical_feature_template.py
│   └── models/
│
├── detection/
│   └── mri_detection.py
│
├── recommendation/
│
├── llm/
│   ├── llm_interface.py
│   └── safety_guard.py
│
├── models/
│
├── scripts/
│
├── assets/
│
├── requirements.txt
│
├── .env.example
│
└── README.md
```

---

# 🚀 Installation

Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Alzheimers-CDSS.git
```

Navigate

```bash
cd Alzheimers-CDSS
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create

```bash
cp .env.example .env
```

Configure

```
GROQ_API_KEY=YOUR_API_KEY
```

---

# ▶ Running

```bash
python app.py
```

---

# 📊 AI Models

| Model | Purpose |
|---------|----------|
| Random Forest | Clinical Risk Prediction |
| ResNet18 | MRI Classification |
| Llama 3.1 | Explainable AI |
| Rule-Based Engine | Recommendations |

---

# 💼 Engineering Skills Demonstrated

This repository demonstrates practical software engineering and AI development experience in:

- End-to-End Machine Learning Pipelines
- Deep Learning for Medical Imaging
- Explainable AI
- Prompt Engineering
- Computer Vision
- Model Deployment
- Desktop Software Development
- Modular Python Architecture
- Secure Environment Configuration
- Healthcare AI System Design
- Object-Oriented Programming
- Production-Oriented Repository Management

---

# 📈 Future Improvements

- Docker Deployment

- REST API

- FastAPI Backend

- React Frontend

- Grad-CAM Explainability

- Cloud Deployment

- CI/CD Pipeline

- Electronic Health Record Integration

- Multi-modal Deep Learning

---

# ⚠ Disclaimer

This software is intended exclusively for educational and research purposes.

It is **not** a certified medical device and must not be used as a substitute for professional medical diagnosis or treatment.

---

# 👨‍💻 Author

## Md Arshad

**Artificial Intelligence & Machine Learning Engineer**

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Healthcare AI
- Generative AI
- Python Development

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star.

</div>