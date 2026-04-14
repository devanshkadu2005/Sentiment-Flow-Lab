# Sentiment Flow Lab (Real ML Deployment)

This project now uses a two-service architecture so your actual trained ML/DL models are used in production.

1. Next.js frontend on Vercel
2. Python FastAPI model backend on Render or Railway

## Architecture

1. Browser -> Next.js app
2. Next.js API route [app/api/analyze/route.ts](app/api/analyze/route.ts)
3. Python backend [backend/main.py](backend/main.py)
4. Model artifacts from [models](models)

The backend loads and runs these trained models:

1. Naive Bayes (.pkl)
2. SVM (.pkl)
3. LSTM (.h5)
4. CNN-LSTM (.h5)

## Where The Models Are

Model files are expected in [models](models):

1. nb_model.pkl
2. svm_model.pkl
3. tokenizer.pkl
4. metrics.pkl
5. lstm_model.h5
6. cnn_lstm_model.h5

If they do not exist, generate them using [backend/train.py](backend/train.py).

### Reliability Upgrades Implemented

1. Calibrated per-model thresholds are learned and saved during training.
2. Backend applies these thresholds during inference instead of fixed 0.50.
3. Short-text guard marks very short input as low-reliability.
4. Exact 2/4 split is treated as non-deterministic.

## Local Run (End-to-End)

You can use either backend-prefixed commands or root compatibility commands.

Compatibility commands from project root:

```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

1. Train models (once)

```bash
.\.venv310\Scripts\python.exe backend/train.py
```

Training defaults to full IMDB split. For faster experiments:

```bash
set FAST_DEV=1
.\.venv310\Scripts\python.exe backend/train.py
```

2. Start Python backend

```bash
.\.venv310\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv310\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

3. Configure frontend env

```bash
copy .env.example .env.local
```

Set in `.env.local`:

```bash
PYTHON_BACKEND_URL=http://127.0.0.1:8000
```

4. Start frontend

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Deploy To Production

### A) Deploy Python backend (Render or Railway)

1. Create a new web service from this repo, using [backend](backend) as root.
2. Build command:

```bash
pip install -r requirements.txt
```

3. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

4. Set `MODELS_DIR` if your artifacts are mounted elsewhere.
5. Ensure model files are present at deploy time.

### B) Deploy frontend on Vercel

1. Import the same repository in Vercel.
2. Add env variable in Vercel project settings:

```bash
PYTHON_BACKEND_URL=https://your-backend-service-url
```

3. Deploy.

## Important Notes

1. Vercel hosts the web layer, not TensorFlow-heavy Python inference.
2. Keep model artifacts with backend hosting, not in frontend runtime.
3. If model files are too large for regular git, use Git LFS or object storage.
