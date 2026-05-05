## Architecture

This repository contains a lightweight, educational web application that guides users through a **7-step ML learning pipeline** across multiple clinical specialties. The system is intentionally simple: a **static frontend** orchestrates the workflow and calls a **Python FastAPI backend** for data preparation operations.

---

## High-level goals

- Provide a **step-by-step** learning experience (clinical context → data exploration → preparation → modeling → results → explainability → ethics).
- Keep setup friction low: **no build tools**, no database, no authentication.
- Perform only the compute-heavy / library-heavy steps (imputation, scaling, SMOTE, splitting) in a backend API.

---

## Tech Stack
| Layer | Technology | Reason |
|-------|------------|--------|
| Frontend | HTML/CSS/JS | Required toolchain; fast HMR, component-based UI suits 7-step pipeline |
| Backend | FastAPI (Python) | Required toolchain; auto-generates /docs endpoint, async support, easy scikit-learn integration |
| ML Engine | scikit-learn | Required toolchain; all 6 required models available out of the box |
| Storage | Browser Session Storage | No database needed; state clears on tab close, zero backend storage cost |

## System overview

### Components

- **Frontend (static HTML/CSS/JS)**: `frontend/`
  - Multi-page flow using `step*.html` pages.
  - Loads bundled datasets (CSV) from `frontend/datasets2/`.
  - Collects user choices (target column, feature roles, preprocessing settings).
  - Calls backend API and renders “before/after” summaries.

- **Backend (FastAPI)**: `backend/`
  - Exposes API endpoints, validates requests, and orchestrates pipeline flow.
  - Handles HTTP concerns (routing, CORS, response shaping, result storage, PDF/AI-advisor integration).
  - Delegates core preprocessing/training/evaluation/explainability logic to the `ml/` package.

- **ML workspace**: `ml/`
  - Contains reusable ML core modules used by backend routers.
  - `ml/preprocessing/`: outlier detection/filtering, imputation, normalization, SMOTE.
  - `ml/models/`: model creation/training, metrics/evaluation, explainability and fairness helpers.
  - Keeps ML logic modular so backend remains API-focused.

- **Docs**: `docs/`
  - Run instructions 
  - Architecture stack 
  - Definition of Done
  - Backlog Prioritization
---

## Repository layout

```text
HealthAi_juniorEngineers/
├── backend/
├── frontend/
├── ml/
└── docs/
```

---

##  API Endpoints Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/prepare` | Prepare data (column handling, split, imputation, normalization, optional SMOTE/outlier removal) |
| `POST` | `/api/train` | Train model and return metrics, explainability, and fairness payload |
| `GET` | `/api/results/{result_id}` | Fetch stored summary metrics for a previous training result |
| `POST` | `/api/detect-outliers` | Detect outliers using IQR-based analysis |
| `POST` | `/api/generate-certificate` | Generate PDF certificate/summary report |
| `POST` | `/api/ai-advisor` | Stream LLM-based clinical model recommendation |
| `GET` | `/api/docs` | Auto-generated FastAPI documentation |
| `GET` | `/health` | Lightweight service health check |
---

## Project management toolchain

| Tool | Category | Purpose |
|---|---|---|
| Jira | Project Management | Product backlog, sprint backlog, user stories, story points, velocity tracking, burndown charts |
| GitHub | Version Control | All source code, feature branches, pull requests, code review |
| GitHub Wiki | Documentation | Architecture decisions, meeting notes, retrospective boards, API docs, sprint notes |
| Figma | UI/UX Design | Wireframes and high-fidelity mockups for all 7 steps; clickable prototype |
| Miro | Retrospectives | Sprint retrospective boards — Keep / Improve / Try format |
| Google Forms / Maze | User Testing | Usability testing with non-CS participants (Weeks 10-11) |

---

## Privacy & Security

- All communication over HTTPS
- No personal data stored in the browser beyond the current session
- Uploaded CSV data is held temporarily in Browser Session Storage and cleared when the tab is closed
- PII filtering and anonymization handled server-side before any ML processing
- No user data is persisted on the backend

---

## Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready, protected |
| `feature/sprint[N]-[desc]` | Feature development |
| `bugfix/[desc]` | Bug fixes |
| `hotfix/[desc]` | Urgent fixes |

- All merges to `main` require **1 PR approval** from the Lead Developer.
