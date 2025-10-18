import asyncio
import json
import logging
import os
import hashlib
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:maramdaas@localhost/problem_aggregator"

import asyncpg

def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = ' '.join(text.split())
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    words = preprocess_text(text).split()
    
    simple_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    words = [word for word in words if word not in simple_stopwords and len(word) > 2]
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(max_keywords)]
    return keywords

def generate_simple_solution(problem_text: str) -> str:
    keywords = extract_keywords(problem_text, 5)
    
    solutions = {
        'money': "Consider creating a budget, exploring additional income sources, or seeking financial advice.",
        'work': "Try discussing with your supervisor, updating your skills, or exploring new opportunities.",
        'health': "Consult with a healthcare professional and consider lifestyle changes.",
        'relationship': "Open communication and perhaps couples counseling could help.",
        'stress': "Practice stress management techniques like meditation or exercise.",
        'time': "Try time management techniques like prioritization and scheduling.",
        'computer': "Check online tutorials, update software, or consult tech support.",
        'family': "Consider family counseling or open dialogue with family members.",
    }
    
    for keyword in keywords:
        for category, solution in solutions.items():
            if category in keyword or keyword in category:
                return solution
    
    return "Break down the problem into smaller steps, research possible solutions, and consider asking for help from experts or friends."

class ProblemSubmission(BaseModel):
    text: str

class Problem(BaseModel):
    id: int
    text: str
    keywords: List[str]
    text_hash: str
    created_at: str

class Cluster(BaseModel):
    id: int
    representative_text: str
    solution: str
    count: int
    problems: List[str] = []
    keywords: List[str] = []

async def init_database():
    conn = await asyncpg.connect(DATABASE_URL)
    
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS problems (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            keywords TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS clusters (
            id SERIAL PRIMARY KEY,
            representative_text TEXT NOT NULL,
            solution TEXT NOT NULL,
            count INTEGER NOT NULL,
            problems TEXT NOT NULL,
            keywords TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    await conn.close()
    logger.info("Database initialized")

async def store_problem(text: str) -> int:
    text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
    keywords = extract_keywords(text)
    
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS problems (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                keywords TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        problem_id = await conn.fetchval(
            "INSERT INTO problems (text, keywords, text_hash) VALUES ($1, $2, $3) RETURNING id",
            text, json.dumps(keywords), text_hash
        )
        logger.info(f"Stored problem {problem_id}: {text[:50]}...")
        return problem_id
    finally:
        await conn.close()

async def get_all_problems() -> List[Problem]:
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch("SELECT id, text, keywords, text_hash, created_at FROM problems ORDER BY created_at DESC")
    await conn.close()
    
    problems = []
    for row in rows:
        problems.append(Problem(
            id=row['id'],
            text=row['text'],
            keywords=json.loads(row['keywords']),
            text_hash=row['text_hash'],
            created_at=str(row['created_at'])
        ))
    
    return problems

async def store_clusters(clusters_data: List[dict]):
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("DELETE FROM clusters")
    
    for cluster in clusters_data:
        await conn.execute(
            "INSERT INTO clusters (representative_text, solution, count, problems, keywords) VALUES ($1, $2, $3, $4, $5)",
            cluster['representative_text'],
            cluster['solution'],
            cluster['count'],
            json.dumps(cluster['problems']),
            json.dumps(cluster['keywords'])
        )
    
    await conn.close()
    logger.info(f"Stored {len(clusters_data)} clusters")

async def get_clusters() -> List[Cluster]:
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch("SELECT * FROM clusters ORDER BY count DESC")
    await conn.close()
    
    clusters = []
    for row in rows:
        clusters.append(Cluster(
            id=row['id'],
            representative_text=row['representative_text'],
            solution=row['solution'],
            count=row['count'],
            problems=json.loads(row['problems']),
            keywords=json.loads(row['keywords'])
        ))
    
    return clusters

def perform_lightweight_clustering(problems: List[Problem], max_clusters: int = 15) -> List[dict]:
    if len(problems) < 2:
        return []
    
    texts = [p.text for p in problems]
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.9
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        logger.warning("TF-IDF failed, using keyword-based clustering")
        return keyword_based_clustering(problems)
    
    n_problems = len(problems)
    n_clusters = min(max_clusters, max(2, n_problems // 3))
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=50,
        n_init=5,
        init='k-means++'
    )
    
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(problems[i])
    
    cluster_data = []
    for label, cluster_problems in clusters.items():
        if len(cluster_problems) < 1:
            continue
        
        representative_problem = max(cluster_problems, key=lambda p: len(p.text))
        
        all_keywords = []
        for p in cluster_problems:
            all_keywords.extend(p.keywords)
        
        keyword_counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counter.most_common(8)]
        
        solution = generate_simple_solution(representative_problem.text)
        
        cluster_data.append({
            'representative_text': representative_problem.text,
            'solution': solution,
            'count': len(cluster_problems),
            'problems': [p.text for p in cluster_problems],
            'keywords': top_keywords
        })
    
    cluster_data.sort(key=lambda x: x['count'], reverse=True)
    
    logger.info(f"Generated {len(cluster_data)} clusters from {len(problems)} problems")
    return cluster_data

def keyword_based_clustering(problems: List[Problem]) -> List[dict]:
    clusters = {}
    
    for problem in problems:
        best_cluster = None
        best_score = 0
        
        for cluster_id, cluster_problems in clusters.items():
            cluster_keywords = set()
            for p in cluster_problems:
                cluster_keywords.update(p.keywords)
            
            problem_keywords = set(problem.keywords)
            intersection = len(cluster_keywords & problem_keywords)
            union = len(cluster_keywords | problem_keywords)
            
            if union > 0:
                similarity = intersection / union
                if similarity > best_score and similarity > 0.2:
                    best_score = similarity
                    best_cluster = cluster_id
        
        if best_cluster is not None:
            clusters[best_cluster].append(problem)
        else:
            new_id = len(clusters)
            clusters[new_id] = [problem]
    
    cluster_data = []
    for cluster_id, cluster_problems in clusters.items():
        if len(cluster_problems) < 2:
            continue
        
        representative_problem = max(cluster_problems, key=lambda p: len(p.text))
        all_keywords = []
        for p in cluster_problems:
            all_keywords.extend(p.keywords)
        
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]
        solution = generate_simple_solution(representative_problem.text)
        
        cluster_data.append({
            'representative_text': representative_problem.text,
            'solution': solution,
            'count': len(cluster_problems),
            'problems': [p.text for p in cluster_problems],
            'keywords': top_keywords
        })
    
    return cluster_data

app = FastAPI(title="Lightweight Problem Aggregator", description="Submit and cluster problems efficiently")
security = HTTPBasic()

ADMIN_PASSWORD = "maramdaas"

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

MAIN_PAGE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Aggregator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h2 class="text-center mb-0">Problem Aggregator</h2>
                        <p class="text-center text-muted mt-2">Submit your problem and we'll help find solutions</p>
                    </div>
                    <div class="card-body">
                        {% if success %}
                        <div class="alert alert-success">
                            Problem submitted successfully! Thank you for sharing.
                        </div>
                        {% endif %}
                        
                        <form method="post" action="/">
                            <div class="mb-3">
                                <label for="problem_text" class="form-label">Describe your problem:</label>
                                <textarea 
                                    class="form-control" 
                                    id="problem_text" 
                                    name="problem_text" 
                                    rows="4" 
                                    placeholder="Tell us about the challenge you're facing..."
                                    required
                                    maxlength="200"
                                ></textarea>
                                <div class="form-text">Maximum 200 characters</div>
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Submit Problem</button>
                        </form>
                        
                        <hr class="mt-4">
                        <div class="text-center">
                            <a href="/admin" class="btn btn-outline-secondary">Admin Dashboard</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

ADMIN_PAGE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Problem Aggregator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid mt-3">
        <div class="row">
            <div class="col-12">
                <h1>Admin Dashboard</h1>
                <p class="text-muted">Manage problems and view solutions (Lightweight Version)</p>
                
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card bg-primary text-white">
                            <div class="card-body">
                                <h5>Total Problems</h5>
                                <h2>{{ total_problems }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-success text-white">
                            <div class="card-body">
                                <h5>Problem Clusters</h5>
                                <h2>{{ total_clusters }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5>Actions</h5>
                                <form method="post" action="/admin/cluster" class="d-inline">
                                    <button type="submit" class="btn btn-warning me-2">Run Clustering</button>
                                </form>
                                <a href="/" class="btn btn-outline-primary">Back to Submission</a>
                                <button onclick="location.reload()" class="btn btn-outline-secondary">Refresh</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                {% if clusters %}
                <div class="card">
                    <div class="card-header">
                        <h3>Problem Clusters & Solutions</h3>
                        <small class="text-muted">Using lightweight TF-IDF clustering</small>
                    </div>
                    <div class="card-body">
                        {% for cluster in clusters %}
                        <div class="card mb-3">
                            <div class="card-header d-flex justify-content-between">
                                <h5>Cluster {{ loop.index }}</h5>
                                <span class="badge bg-primary">{{ cluster.count }} reports</span>
                            </div>
                            <div class="card-body">
                                <h6>Representative Problem:</h6>
                                <p class="text-muted">{{ cluster.representative_text }}</p>
                                
                                <h6>Key Topics:</h6>
                                <div class="mb-2">
                                    {% for keyword in cluster.keywords[:5] %}
                                    <span class="badge bg-secondary me-1">{{ keyword }}</span>
                                    {% endfor %}
                                </div>
                                
                                <h6>Suggested Solution:</h6>
                                <div class="alert alert-info">{{ cluster.solution }}</div>
                                
                                <details>
                                    <summary class="btn btn-sm btn-outline-secondary">View all {{ cluster.count }} similar problems</summary>
                                    <div class="mt-2">
                                        <ul class="list-group list-group-flush">
                                            {% for problem in cluster.problems %}
                                            <li class="list-group-item">{{ problem }}</li>
                                            {% endfor %}
                                        </ul>
                                    </div>
                                </details>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% else %}
                <div class="alert alert-info">
                    <h4>No clusters available</h4>
                    <p>Click "Run Clustering" to analyze submitted problems and generate solutions.</p>
                </div>
                {% endif %}
                
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request, success: bool = False, duplicate: bool = False):
    from jinja2 import Template
    template = Template(MAIN_PAGE_HTML)
    html_content = template.render(success=success, duplicate=duplicate)
    return HTMLResponse(content=html_content)

@app.post("/", response_class=HTMLResponse)
async def submit_problem(request: Request, problem_text: str = Form(...)):
    error_message = None
    
    if len(problem_text.strip()) < 10:
        error_message = "Problem description is too short. Please provide at least 10 characters."
    elif len(problem_text) > 200:
        error_message = "Problem description is too long. Maximum 200 characters allowed."
    
    if error_message:
        from jinja2 import Template
        template = Template(MAIN_PAGE_HTML.replace(
            '{% endif %}',
            '{% endif %}\n                        {% if error %}\n                        <div class="alert alert-danger">\n                            {{ error }}\n                        </div>\n                        {% endif %}'
        ))
        html_content = template.render(error=error_message)
        return HTMLResponse(content=html_content)
    
    problem_id = await store_problem(problem_text.strip())
    
    from jinja2 import Template
    template = Template(MAIN_PAGE_HTML)
    html_content = template.render(success=True)
    return HTMLResponse(content=html_content)

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    problems = await get_all_problems()
    clusters = await get_clusters()
    
    from jinja2 import Template
    template = Template(ADMIN_PAGE_HTML)
    html_content = template.render(
        total_problems=len(problems),
        total_clusters=len(clusters),
        clusters=clusters
    )
    return HTMLResponse(content=html_content)

@app.post("/admin/cluster", response_class=HTMLResponse)
async def trigger_clustering(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    problems = await get_all_problems()
    
    if len(problems) < 2:
        clusters = await get_clusters()
        from jinja2 import Template
        template = Template(ADMIN_PAGE_HTML.replace(
            '<h5>Actions</h5>',
            '<h5>Actions</h5><div class="alert alert-warning">Need at least 2 problems to perform clustering</div>'
        ))
        html_content = template.render(
            total_problems=len(problems),
            total_clusters=len(clusters),
            clusters=clusters
        )
        return HTMLResponse(content=html_content)
    
    clusters_data = perform_lightweight_clustering(problems)
    await store_clusters(clusters_data)
    
    clusters = await get_clusters()
    from jinja2 import Template
    template = Template(ADMIN_PAGE_HTML.replace(
        '<h5>Actions</h5>',
        f'<h5>Actions</h5><div class="alert alert-success">Clustering completed! Generated {len(clusters_data)} clusters.</div>'
    ))
    html_content = template.render(
        total_problems=len(problems),
        total_clusters=len(clusters),
        clusters=clusters
    )
    return HTMLResponse(content=html_content)

@app.get("/api/problems")
async def get_problems():
    problems = await get_all_problems()
    return [{"id": p.id, "text": p.text, "keywords": p.keywords, "created_at": p.created_at} for p in problems]

@app.get("/api/clusters")
async def get_clusters_api():
    clusters = await get_clusters()
    return [{"id": c.id, "representative_text": c.representative_text, "solution": c.solution, "count": c.count, "keywords": c.keywords} for c in clusters]

@app.on_event("startup")
async def startup_event():
    await init_database()
    logger.info("Lightweight Problem Aggregator started successfully!")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)