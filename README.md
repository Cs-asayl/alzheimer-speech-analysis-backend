# Alzheimer Speech Analysis Backend

Python-based AI backend for the **Alzheimer Speech Assessment** mobile application.

This backend processes speech recordings collected by the Flutter mobile application and performs acoustic and linguistic analysis for Alzheimer's disease risk screening.

The system combines **88 eGeMAPS acoustic features** with **19 linguistic features**, producing **107 features**. After feature scaling and Genetic Algorithm (GA) feature selection, **56 selected features** are passed to a Logistic Regression classifier.

> **Important:** This project is a research prototype intended for screening and educational purposes only. It is not a medical diagnostic system.

---

## 🧠 System Overview

The backend acts as the AI processing layer between Firebase Storage and the Flutter mobile application.

```text
Flutter Mobile Application
          │
          │ Voice Recording
          ▼
    Firebase Storage
          │
          │ Storage Path
          ▼
     Flask REST API
          │
          ├──────────────────────┐
          ▼                      ▼
   Acoustic Analysis      Speech-to-Text
          │                      │
          ▼                      ▼
  88 eGeMAPS Features    19 Linguistic Features
          │                      │
          └──────────┬───────────┘
                     ▼
             107 Combined Features
                     │
                     ▼
                StandardScaler
                     │
                     ▼
          Genetic Algorithm (GA)
                     │
                     ▼
             56 Selected Features
                     │
                     ▼
           Logistic Regression
                     │
                     ▼
          Healthy / Patient
              Classification
                     │
                     ▼
              Flask Response
                     │
                     ▼
          Flutter Result Screen
```

---

## ✨ Features

- Flask REST API
- Firebase Storage integration
- Automatic audio retrieval from Firebase
- eGeMAPS acoustic feature extraction
- Speech-to-Text processing
- Linguistic feature extraction
- Acoustic and linguistic feature fusion
- Feature scaling
- Genetic Algorithm feature selection
- Logistic Regression classification
- Healthy / Patient prediction
- Probability-based prediction output
- Integration with the Flutter mobile application

---

## 🔬 Feature Extraction

### Acoustic Features

The backend extracts **88 eGeMAPS acoustic features** from each speech recording.

These features represent characteristics of the speech signal and voice and form the acoustic component of the analysis pipeline.

```text
Audio Recording
      ↓
eGeMAPS Extraction
      ↓
88 Acoustic Features
```

### Linguistic Features

Speech is processed through the Speech-to-Text component before linguistic feature extraction.

The system extracts **19 linguistic features** representing characteristics of the transcribed speech.

```text
Audio Recording
      ↓
Speech-to-Text
      ↓
Transcript
      ↓
Linguistic Analysis
      ↓
19 Linguistic Features
```

### Combined Feature Vector

The two feature groups are combined:

```text
88 Acoustic Features
        +
19 Linguistic Features
        =
107 Total Features
```

---

## 🤖 Machine Learning Pipeline

The complete feature vector contains **107 features**.

The machine learning pipeline performs:

1. Feature combination
2. Feature scaling using the trained scaler
3. Genetic Algorithm feature selection
4. Reduction from 107 to **56 selected features**
5. Logistic Regression classification
6. Healthy / Patient probability calculation
7. Risk result generation

The trained model artifacts are stored inside the `models/` directory.

---

## 📊 Model Files

```text
models/
│
├── eGeMAPS_Linguistic_Scaler.pkl
├── GA_LogisticRegression.pkl
├── GA_Selected_Features.json
└── GA_Selected_Indices.pkl
```

### Model Components

**eGeMAPS_Linguistic_Scaler.pkl**  
Stores the scaler used to transform the 107-feature input vector.

**GA_Selected_Indices.pkl**  
Stores the indices of features selected by the Genetic Algorithm.

**GA_Selected_Features.json**  
Stores information about the selected features.

**GA_LogisticRegression.pkl**  
Stores the trained Logistic Regression classifier used for final prediction.

---

## 🌐 REST API

### Health Check

```http
GET /
```

Used to verify that the Flask server is running.

Example response:

```json
{
  "message": "Alzheimer Speech Analysis API is running"
}
```

---

### Analyze Speech

```http
POST /analyze
```

The Flutter application sends the Firebase Storage path of a recorded audio file.

Example request:

```json
{
  "storagePath": "users/user_id/audio/recording.m4a"
}
```

The backend then:

1. Retrieves the audio file from Firebase Storage.
2. Extracts acoustic features.
3. Processes speech for linguistic analysis.
4. Combines the features.
5. Applies the trained machine learning pipeline.
6. Generates prediction probabilities.
7. Returns the result to Flutter.

A prediction response contains the classification, probability information, and risk result used by the mobile application's result screen.

---

## 📂 Project Structure

```text
alzheimer_backend/
│
├── app.py
├── feature_extractor.py
├── firebase_service.py
├── linguistic_extractor.py
├── model_predictor.py
├── speech_to_text.py
├── test_models.py
│
├── models/
│   ├── eGeMAPS_Linguistic_Scaler.pkl
│   ├── GA_LogisticRegression.pkl
│   ├── GA_Selected_Features.json
│   └── GA_Selected_Indices.pkl
│
├── .gitignore
└── README.md
```

### Main Files

`app.py`  
Runs the Flask API and coordinates the speech-analysis pipeline.

`firebase_service.py`  
Handles audio retrieval from Firebase Storage.

`feature_extractor.py`  
Extracts eGeMAPS acoustic features from speech recordings.

`speech_to_text.py`  
Handles Speech-to-Text processing.

`linguistic_extractor.py`  
Extracts linguistic features from transcribed speech.

`model_predictor.py`  
Loads the scaler, GA-selected features, and Logistic Regression model and performs prediction.

`test_models.py`  
Provides model/pipeline testing functionality.

---

## 🛠️ Technologies Used

### Backend

- Python
- Flask
- REST API

### Artificial Intelligence

- Machine Learning
- Logistic Regression
- Genetic Algorithm
- Feature Scaling

### Speech Processing

- eGeMAPS
- Speech-to-Text
- Acoustic Feature Extraction
- Linguistic Feature Extraction

### Cloud Integration

- Firebase Storage
- Firebase Admin SDK

### Mobile Integration

- Flutter
- Dart

---

## 🚀 Running the Backend

### 1. Clone the Repository

```bash
git clone https://github.com/Cs-asayl/alzheimer-speech-analysis-backend.git
```

Navigate to the project:

```bash
cd alzheimer-speech-analysis-backend
```

### 2. Install Required Dependencies

Install the Python packages required by the backend.

> The exact dependency list should match the Python environment used to run this project.

### 3. Configure Firebase

The backend requires Firebase Admin credentials to access Firebase Storage.

For security reasons, the Firebase service account credential file is **not included in this repository**.

Configure your own Firebase project and service account credentials before running the server.

Never commit private Firebase service-account credentials to a public repository.

### 4. Start the Flask Server

```bash
python app.py
```

The development server runs on port:

```text
5000
```

During local development, the Flutter application must use an address that can reach the Flask server.

---

## 📱 Mobile Application

The Flutter mobile application provides the user interface for authentication, voice recording, assessments, and result visualization.

**Mobile repository:**  
[Alzheimer Speech Assessment Mobile App](https://github.com/Cs-asayl/alzheimer-speech-assessment-mobile)

---

## 🔄 End-to-End Workflow

```text
User
 ↓
Flutter Application
 ↓
Voice Assessment
 ↓
Audio Recording
 ↓
Firebase Storage
 ↓
Flask Backend
 ↓
Acoustic + Linguistic Analysis
 ↓
107 Features
 ↓
Feature Scaling
 ↓
GA Feature Selection
 ↓
56 Features
 ↓
Logistic Regression
 ↓
Prediction Probabilities
 ↓
Flutter Result Screen
```

---

## 🎯 Project Purpose

The project explores the use of **speech analysis and machine learning** as a non-invasive approach to support early Alzheimer's disease risk screening.

It combines a mobile speech-collection interface with an automated AI pipeline capable of analyzing acoustic and linguistic characteristics of speech.

---

## ⚕️ Medical Disclaimer

This project was developed for **research and educational purposes**.

The Healthy / Patient classification generated by the machine learning model represents a research screening output and **must not be interpreted as a clinical diagnosis**.

Diagnosis of Alzheimer's disease or other cognitive conditions must be performed by qualified healthcare professionals.