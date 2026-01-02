# Network Intrusion Detection Using Machine Learning

This project evaluates multiple machine learning models for detecting malicious network traffic using the UNSW-NB15 dataset.

## Research Paper
I have written a research paper summarizing the methodology, experiments, and findings of this project. The paper is included in this repository as a PDF under Intrusion_Detection_Paper.pdf

## Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

## Key Features
- Data preprocessing and cleaning
- Class imbalance handling using SMOTE
- Model evaluation using F1 score, ROC-AUC, precision-recall curves, and confusion matrices
- Runtime benchmarking for training efficiency
- Automated experiment pipeline in Python

## Results Summary
| Model | F1 Score | ROC-AUC | Training Time |
|------|----------|---------|---------------|
| Logistic Regression | 0.9487 | 0.998 | ~26.6s |
| Random Forest | 0.9831 | 1.000 | ~104s |
| XGBoost | 0.9789 | 1.000 | ~10.2s |
| Neural Network | 0.9600 | 1.000 | ~137s |

Random Forest achieved the highest detection accuracy, while XGBoost offered the best performance-to-runtime tradeoff.

## Dataset
- UNSW-NB15 (Kaggle)

## Technologies
- Python
- scikit-learn
- XGBoost
- TensorFlow / Keras
- SMOTE

## Contact
Natalie Santo

natalierosesanto@gmail.com
