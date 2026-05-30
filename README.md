# HealthCare Connect Clinical AI

Clinical AI service for HealthCare Connect using FastAPI, TensorFlow, and CNN models.

## Overview

This project exposes a trained lung cancer classification model through a REST API.

The API receives a medical image and returns:

- Prediction
- Confidence score
- Risk level
- Clinical recommendation

The service is designed to integrate with:

- Salesforce Health Cloud
- Agentforce
- Flows
- Apex Integrations
- Clinical Triage Systems

---

## Architecture

Patient Image
↓
FastAPI Endpoint
↓
TensorFlow CNN Model
↓
Clinical Prediction
↓
JSON Response
↓
Salesforce Integration

---

## API Endpoint

### Health Check

GET /

Response:

```json
{
  "status": "API running"
}
```

### Prediction

POST /predict

Input:

Medical image file (.jpg, .jpeg, .png)

Response:

```json
{
  "prediction": "Maligno",
  "confidence": 0.9983,
  "normal_probability": 0.0017,
  "benign_probability": 0.0,
  "malignant_probability": 0.9983,
  "risk_level": "High",
  "recommendation": "Specialist review required"
}
```

---

## Local Development

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Cloud Deployment

Platform:

```text
Render
```

Public URL:

```text
https://healthcare-connect-clinical-ai.onrender.com
```

Swagger:

```text
https://healthcare-connect-clinical-ai.onrender.com/docs
```

---

## Technology Stack

- Python 3.12
- FastAPI
- TensorFlow
- NumPy
- Pillow
- Uvicorn
- Render
- GitHub

---

## Future Roadmap

- Salesforce Flow Integration
- Agentforce Actions
- Automated Case Creation
- Clinical Risk Prioritization
- Oncology Escalation Workflows
- Multi-Class Tumor Classification
- Explainable AI (XAI)
- Clinical Decision Support

---

## Author

Emanuel Alejandro Meneses Ramirez

HealthCare Connect Clinical AI
2026
