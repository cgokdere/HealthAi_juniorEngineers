# SETUP.md — HealthAI Junior Engineers

> An interactive web app helping healthcare professionals understand machine learning through a **7-step guided pipeline** across **20 clinical specialties**.

**Course:** SENG 430 · University Project · 2025

---

## 📋 Prerequisites

| Tool | Minimum Version | Check |
|------|----------------|-------|
| Docker & Compose | Latest | `docker compose version` |
| Python | 3.10+ | `python --version` |
| Web Browser | Chrome, Firefox, or Edge | — |
| OS | macOS, Linux, or Windows | — |
| Git | 2.38+ | `git --version` |

---

## 1. Clone the Repository

```bash
git clone https://github.com/cgokdere/HealthAi_juniorEngineers.git
cd HealthAi_juniorEngineers
```

---

## 2. Option 1: Quick Start with Docker (Recommended) 🐳

This is the fastest way to get the entire platform (Backend + Frontend) running in a consistent environment.

```bash
# 1. Start both services
docker-compose up --build

# 2. Access the application
# Frontend: http://localhost:8080
# Backend (API Docs): http://localhost:8000/api/docs
```

---

## 3. Option 2: Manual Setup (Local Development) 🛠️

Use this method if you want to run the components independently or without Docker.

### 3.1 Backend Setup (FastAPI)

1. **Navigate to backend and create virtual environment:**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   python main.py
   # OR
   uvicorn main:app --reload --port 8000
   ```
   ▶ API documented at: **http://localhost:8000/api/docs**

### 3.2 Frontend Setup

The frontend is built with vanilla HTML, CSS, and JS. You need a local server to avoid CORS issues.

1. **Run a basic HTTP server from the project root:**
   ```bash
   # From the root directory (HealthAi_juniorEngineers)
   python3 -m http.server 8080
   ```

2. **Open your browser:**
   ▶ **http://localhost:8080/frontend/index.html**

> [!TIP]
> You can also use the **VS Code Live Server** extension by right-clicking `frontend/index.html` and selecting **Open with Live Server**.

---

## 4. API Endpoints Reference

Once the backend is running, all endpoints are documented interactively at [`/api/docs`](http://localhost:8000/api/docs).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect-outliers` | Detect outliers using IQR method |
| `POST` | `/api/prepare` | Preprocess, normalize, and split data (Step 3-5) |
| `POST` | `/api/train` | Train model and compute metrics/fairness (Step 6) |
| `GET`  | `/api/results/{id}` | Retrieve specific model training results |
| `POST` | `/api/ai-advisor` | Get clinical AI recommendations (Llama-3 via Groq) |
| `GET`  | `/api/certificate` | Generate PDF summary certificate |
| `GET`  | `/api/docs` | Auto-generated FastAPI documentation |

---

## 5. Branch & Workflow Rules

This repository uses branch protection on `main`. **Never push directly to `main`.**

```bash
# 1. Always start from main and pull latest
git checkout main
git pull origin main

# 2. Create your branch following the naming convention
git checkout -b feature/sprint2-data-preprocessing

# 3. Make changes, commit with a clear message
git add .
git commit -m "feat: add CSV preprocessing endpoint"

# 4. Push and open a Pull Request targeting main
git push origin feature/sprint2-data-preprocessing
```

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| New feature | `feature/sprint[N]-[description]` | `feature/sprint2-data-preprocessing` |
| Bug fix | `bugfix/[description]` | `bugfix/csv-upload-crash` |
| Urgent fix | `hotfix/[description]` | `hotfix/model-timeout` |

**Merge Rules:**
- At least **1 reviewer approval** is required before merging
- CI checks must pass before merge
- Delete your branch after merging

---

## 🆘 Need Help?

- Browse open [Issues](https://github.com/cgokdere/HealthAi_juniorEngineers/issues) on GitHub
- Check the interactive API docs at [https://healthai-juniorengineers-1.onrender.com/api/docs](https://healthai-juniorengineers-1.onrender.com/api/docs)
- Review architecture diagrams and sprint notes in the `/docs` folder
