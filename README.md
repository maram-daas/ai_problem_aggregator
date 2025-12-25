
# AI Problem Aggregator (TinyLlama)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)

A lightweight **AI-powered web application** that collects user-submitted problems, **clusters semantically similar issues using a local LLM (TinyLlama via Ollama)**, and generates **automated, human-readable solutions** — fully offline and free.

---

## 🚀 Key Highlights

* ✅ **Runs fully locally** (no OpenAI / paid APIs)
* 🤖 **AI-powered semantic clustering** using TinyLlama
* 🧠 **Natural-language solutions** generated per cluster
* 🔐 **Admin dashboard** with secure access
* ⚡ **FastAPI + async PostgreSQL**
* 🆓 100% free, open-source stack

---

## ✨ Features

* **Problem Submission**

  * Simple public form
  * 10–200 character limit
* **AI Semantic Clustering**

  * Uses **TinyLlama** via **Ollama**
  * Groups problems by meaning, not keywords
  * Intelligent fallbacks if AI output fails
* **AI-Generated Solutions**

  * Short, actionable advice per cluster
  * Root-cause–focused responses
* **Admin Dashboard**

  * View all problems
  * Trigger AI clustering
  * Inspect clusters and solutions
* **REST API**

  * Access problems and clusters programmatically

---

## 🧱 Tech Stack

### Backend

* **FastAPI**
* **Uvicorn**
* **AsyncPG**
* **PostgreSQL**

### AI / NLP

* **Ollama** (local LLM runtime)
* **TinyLlama** (lightweight language model)

### Frontend

* **Jinja2 templates**
* **Vanilla HTML/CSS (no JS frameworks)**

---

## 📦 Prerequisites

* Python **3.8+**
* PostgreSQL **12+**
* **Ollama installed locally**

---

## 🧠 Installing Ollama (Required)

This project **requires Ollama** to run AI clustering.

### 1️⃣ Install Ollama

👉 [https://ollama.com/download](https://ollama.com/download)

Verify installation:

```bash
ollama --version
```

### 2️⃣ Start Ollama Server

```bash
ollama serve
```

### 3️⃣ Pull TinyLlama Model

```bash
ollama pull tinyllama
```

> ℹ️ TinyLlama is fast, lightweight, and ideal for local AI workloads.

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/maram-daas/ai_problem_aggregator.git
cd ai_problem_aggregator
```

---

### 2. Set Up PostgreSQL Database

```bash
psql -U postgres
CREATE DATABASE problem_aggregator;
\q
```

Update database credentials in `problem_aggregator.py`:

```python
DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost/problem_aggregator"
```

---

### 3. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python problem_aggregator.py
```

The app runs at:

```
http://127.0.0.1:8001
```

---

## 🌐 Access Points

| Feature           | URL                                                                      |
| ----------------- | ------------------------------------------------------------------------ |
| Public Submission | [http://127.0.0.1:8001](http://127.0.0.1:8001)                           |
| Admin Dashboard   | [http://127.0.0.1:8001/admin](http://127.0.0.1:8001/admin)               |

### 🔐 Admin Login

* **Username**: anything
* **Password**: `maramdaas`

---

## 🧪 Sample Problems to Test Clustering

Submit at least **2 problems**:

Then click **“Run AI Clustering”** in the admin dashboard.

---

## 🧠 How It Works

1. **User submits problem**
2. **Problems stored in PostgreSQL**
3. **Admin triggers clustering**
4. **TinyLlama analyzes semantic meaning**
5. **Problems grouped into clusters**
6. **AI generates a concise solution per cluster**
7. **Results displayed in admin UI**

Fallback strategies ensure clustering still works even if AI output is imperfect.

---

## 🐛 Troubleshooting

### ❌ Ollama Connection Error

```
Cannot connect to Ollama
```

✅ Fix:

```bash
ollama serve
```

---

### ❌ No AI Clusters Generated

* Ensure `tinyllama` is pulled
* Submit at least **2 problems**

---

### ❌ Database Errors

```
asyncpg.exceptions.InvalidPasswordError
```

✅ Fix: Update `DATABASE_URL`

---

## 📄 License

MIT License

---

## 🤝 Contributing

This is an experimental learning project I built in my free time to explore how AI can be integrated into a real backend system. It is not optimized and is intended purely for learning purposes.

Feedback and suggestions are highly appreciated. <3

---
