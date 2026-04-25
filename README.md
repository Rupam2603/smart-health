<<<<<<< HEAD
# The Smart Health Assistant 🏥

An AI-driven healthcare support platform that integrates machine learning, real-time APIs, and an intuitive user interface.

## Features
- **Health Prediction**: ML-powered assessment for conditions like diabetes using clinical parameters.
- **Symptom Checker**: NLP-driven triage for common health concerns.
- **Hospital Discovery**: Location-based discovery of nearby medical centers.
- **Vitals Dashboard**: Real-time tracking of heart rate, BP, and more.
- **Emergency SOS**: One-tap emergency response protocol.

## Tech Stack
- **Frontend**: React.js, Tailwind CSS, Framer Motion
- **Backend**: FastAPI (Python), Uvicorn
- **AI/ML**: Scikit-Learn (Random Forest), NLP Triage Engine

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Node.js & npm

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Machine Learning
The model is pre-trained. If you want to re-train:
```bash
python ml-engine/train_model.py
```

## Environment Variables
Create a `.env` in the `backend/` folder:
```env
GOOGLE_API_KEY=your_key_here
```
=======
# smart-health
>>>>>>> e355d0fda1be681752bce142eade764bb0cdeb19
