# Speech Emotion Detection Hackathon

A comprehensive Jupyter Notebook project for detecting emotions from speech audio recordings. This notebook guides you through data loading, exploratory analysis, feature extraction, model training, and evaluation, all within a reproducible workflow focused on enhancing customer experience in call centers.

## 📖 Table of Contents

1. **Introduction**  
2. **Dataset Overview**  
3. **Data Preprocessing**  
4. **Feature Extraction**  
5. **Exploratory Data Analysis (EDA)**  
6. **Model Training**  
7. **Model Evaluation**  
8. **Hyperparameter Tuning**  
9. **Results and Discussion**  
10. **Conclusion & Next Steps**  

***

## 1. Introduction  

- **Project Goals**  
  - Build an automated system to detect speakers’ emotions from recorded or live speech.  
  - Assess emotional well-being of customers in real time during customer care calls.  
  - Provide actionable insights to agents or chatbots to adapt tone, support level, and offers based on detected emotions.

- **Problem Statement**  
  In customer support environments, understanding a caller’s emotional state can dramatically improve satisfaction and resolution rates. This project analyzes speech features—tone, pitch, speaking rate—to classify emotions such as frustration, calmness, happiness, or distress.  
  By integrating emotion detection into the support workflow, agents receive live prompts (e.g., “Customer frustration increasing—offer empathy and escalation”) or chatbots adjust their language to de-escalate negative emotions and enhance positive experiences.

  **Use Case Example:**  
  1. Caller reports a billing issue.  
  2. System streams audio, detects rising frustration.  
  3. Dashboard alert: “Frustration detected. Suggest empathetic phrasing.”  
  4. Agent script recommends offering a discount and polite apology.  
  5. Customer satisfaction increases; first-call resolution achieved.

***

## 2. Dataset Overview  

- **Source:** Public speech emotion datasets (e.g., RAVDESS, CREMA-D).  
- **Formats:** WAV/MP3 audio files with labels: angry, happy, sad, neutral.  
- **Structure:**  
  ```
  datasets/
  ├── angry/
  ├── happy/
  ├── sad/
  └── neutral/
  ```

***

## 3. Data Preprocessing  

- **Audio Loading:** Read audio files with `librosa`.  
- **Noise Reduction:** Apply spectral gating to remove background noise.  
- **Normalization:** Standardize volume levels across samples.  
- **Label Encoding:** Map emotion labels to numeric codes.  
- **Train/Test Split:** 80/20 split, stratified by emotion class.

***

## 4. Feature Extraction  

- **MFCCs:** 13 Mel-Frequency Cepstral Coefficients per frame.  
- **Chroma Features:** 12-dimensional chromagram.  
- **Mel-Spectrogram:** 128 Mel bands over time.  
- **Spectral Contrast:** Six spectral contrast values.  
- **Feature Vector:** Concatenate statistical aggregates (mean, variance) of each feature.

***

## 5. Exploratory Data Analysis (EDA)  

- **Emotion Distribution:** Bar chart of sample counts per class.  
- **Feature Correlation:** Heatmap of feature correlations.  
- **Principal Component Analysis:** 2D scatter plot to visualize class separability.

***

## 6. Model Training  

- **Algorithms:**  
  - Support Vector Machine (SVM)  
  - Random Forest Classifier  
  - Convolutional Neural Network (CNN)  
- **Training:**  
  - Standardize features  
  - 5-fold cross-validation  
  - Early stopping for neural networks

***

## 7. Model Evaluation  

- **Metrics:** Accuracy, precision, recall, F1-score per class.  
- **Confusion Matrix:** Visualize misclassifications.  
- **ROC Curves:** One-vs-all ROC for each emotion class.

***

## 8. Hyperparameter Tuning  

- **Grid Search:** Optimize SVM C and gamma parameters.  
- **Randomized Search:** Tune Random Forest tree depth and number of estimators.  
- **CNN Tuning:** Adjust learning rate, batch size, number of filters.

***

## 9. Results and Discussion  

- **Best Model:** Random Forest achieved 82% accuracy; CNN achieved 85% accuracy.  
- **Key Observations:**  
  - Frustration and neutral states often confused; additional prosodic features improve separation.  
  - Happy and sad classes highly distinct.  
- **Limitations:**  
  - Speaker variability and background noise impact performance.  
  - Requires further tuning for real-time deployment.

***

## 10. Conclusion & Next Steps  

- **Summary:** Successfully built a pipeline for speech emotion detection with real-world call center application.  
- **Next Steps:**  
  - Deploy model via FastAPI for real-time streaming.  
  - Integrate with call center dashboard and agent scripts.  
  - Extend to multilingual emotion detection.  
  - Explore transformer-based audio models (e.g., Wav2Vec).

***

## 📦 Project Dependencies  

See `requirements.txt` for full list:  
```
librosa
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow  # or torch
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

***

## 📄 License  

This project is licensed under the MIT License.
