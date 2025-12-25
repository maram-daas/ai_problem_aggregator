import asyncio
import json
import logging
from datetime import datetime
from typing import List
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import asyncpg
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:maramdaas@localhost/problem_aggregator"
OLLAMA_URL = "http://localhost:11434/api/generate"

class ProblemSubmission(BaseModel):
    text: str

class Problem(BaseModel):
    id: int
    text: str
    created_at: str

class Cluster(BaseModel):
    id: int
    category: str
    representative_text: str
    solution: str
    count: int
    problems: List[str] = []

async def init_database():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS problems (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS clusters (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL,
            representative_text TEXT NOT NULL,
            solution TEXT NOT NULL,
            count INTEGER NOT NULL,
            problems TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    await conn.close()
    logger.info("Database initialized")

async def store_problem(text: str) -> int:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        problem_id = await conn.fetchval(
            "INSERT INTO problems (text) VALUES ($1) RETURNING id",
            text
        )
        logger.info(f"Stored problem {problem_id}: {text[:50]}...")
        return problem_id
    finally:
        await conn.close()

async def get_all_problems() -> List[Problem]:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch("SELECT id, text, created_at FROM problems ORDER BY created_at DESC")
    finally:
        await conn.close()
    
    problems = []
    for row in rows:
        problems.append(Problem(
            id=row['id'],
            text=row['text'],
            created_at=str(row['created_at'])
        ))
    return problems

async def store_clusters(clusters_data: List[dict]):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("DELETE FROM clusters")
        for cluster in clusters_data:
            await conn.execute(
                "INSERT INTO clusters (category, representative_text, solution, count, problems) VALUES ($1, $2, $3, $4, $5)",
                cluster['category'],
                cluster['representative_text'],
                cluster['solution'],
                cluster['count'],
                json.dumps(cluster['problems'])
            )
        logger.info(f"Stored {len(clusters_data)} clusters")
    finally:
        await conn.close()

async def get_clusters() -> List[Cluster]:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch("SELECT * FROM clusters ORDER BY count DESC")
    finally:
        await conn.close()
    
    clusters = []
    for row in rows:
        clusters.append(Cluster(
            id=row['id'],
            category=row['category'],
            representative_text=row['representative_text'],
            solution=row['solution'],
            count=row['count'],
            problems=json.loads(row['problems'])
        ))
    return clusters

async def call_ollama_simple(prompt: str, model: str = "tinyllama") -> str:  # Default to tinyllama
    """Simple Ollama call that returns just the text response"""
    try:
        logger.info(f"Calling Ollama with model: {model}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        response = await asyncio.to_thread(
            requests.post,
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=120
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama error: {response.status_code} - {response.text}")
            return ""
        
        data = response.json()
        result = data.get('response', '').strip()
        logger.info(f"Ollama response: {result[:200]}...")
        return result
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama! Run: ollama serve")
        return ""
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""

async def ai_cluster_problems(problems: List[Problem]) -> List[dict]:
    """Smarter AI-powered clustering using semantic analysis"""
    if not problems:
        return []

    logger.info(f"Starting AI clustering on {len(problems)} problems")
    
    # Use batch processing for efficiency
    problem_texts = [p.text for p in problems]
    
    # Try AI clustering first
    clusters_data = await intelligent_batch_clustering(problem_texts)
    
    # If AI clustering fails or produces poor results, use enhanced fallback
    if not clusters_data or len(clusters_data) == 0:
        logger.warning("AI clustering failed, using enhanced semantic fallback")
        clusters_data = await semantic_fallback_clustering(problem_texts)
    
    # Store and return results
    if clusters_data:
        await store_clusters(clusters_data)
    
    return clusters_data

async def intelligent_batch_clustering(problem_texts: List[str]) -> List[dict]:
    """Advanced clustering using AI to find semantic patterns"""
    try:
        # Prepare problems for batch analysis
        problems_formatted = "\n".join(
            f"PROBLEM_{i}: {text}" 
            for i, text in enumerate(problem_texts[:20])  # Reduced for tinyllama
        )
        
        prompt = f"""
Group these problems into clusters based on similar issues:

{problems_formatted}

For each cluster, give me:
1. Category name (2-3 words)
2. Problem numbers in this cluster (like PROBLEM_0, PROBLEM_1)
3. One example problem

Format as JSON with "clusters" array.
"""
        
        # FIX: Use tinyllama instead of llama2
        response = await call_ollama_simple(prompt, model="tinyllama")  # Changed here
        
        if not response:
            return []
        
        # Try to extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            # If no JSON, try to parse as text
            return await parse_text_response(response, problem_texts)
        
        json_str = response[json_start:json_end]
        
        try:
            result = json.loads(json_str)
            clusters = result.get("clusters", [])
            
            # Convert to our format
            clusters_data = []
            for cluster in clusters:
                actual_problems = []
                for prob_id in cluster.get("problem_ids", []):
                    try:
                        idx = int(prob_id.split("_")[1])
                        if idx < len(problem_texts):
                            actual_problems.append(problem_texts[idx])
                    except (IndexError, ValueError):
                        continue
                
                if actual_problems:
                    clusters_data.append({
                        "category": cluster.get("category", "Uncategorized"),
                        "representative_text": cluster.get("representative_text", actual_problems[0]),
                        "solution": "",  # Will generate separately
                        "count": len(actual_problems),
                        "problems": actual_problems
                    })
            
            logger.info(f"AI generated {len(clusters_data)} clusters")
            return clusters_data
            
        except json.JSONDecodeError:
            # Fallback to text parsing
            return await parse_text_response(response, problem_texts)
            
    except Exception as e:
        logger.error(f"Error in intelligent clustering: {e}")
        return []

async def parse_text_response(response: str, problem_texts: List[str]) -> List[dict]:
    """Parse non-JSON response from tinyllama"""
    # Simple keyword-based clustering for tinyllama
    financial_keywords = ['broke', 'money', 'financial', 'struggling', 'poor', 'cash']
    abuse_keywords = ['hits', 'abuse', 'violence', 'hurt', 'father', 'mother', 'parent']
    technical_keywords = ['error', 'bug', 'crash', 'broken', 'not working']
    
    clusters_map = {
        "Financial Issues": [],
        "Abuse/Physical Issues": [], 
        "Technical Problems": [],
        "General Issues": []
    }
    
    for i, text in enumerate(problem_texts):
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in financial_keywords):
            clusters_map["Financial Issues"].append(text)
        elif any(keyword in text_lower for keyword in abuse_keywords):
            clusters_map["Abuse/Physical Issues"].append(text)
        elif any(keyword in text_lower for keyword in technical_keywords):
            clusters_map["Technical Problems"].append(text)
        else:
            clusters_map["General Issues"].append(text)
    
    # Remove empty clusters
    clusters_data = []
    for category, problems in clusters_map.items():
        if problems:
            clusters_data.append({
                "category": category,
                "representative_text": problems[0],
                "solution": "",
                "count": len(problems),
                "problems": problems
            })
    
    return clusters_data

# Also update the semantic fallback to use tinyllama
async def semantic_fallback_clustering(problem_texts: List[str]) -> List[dict]:
    """Enhanced fallback using tinyllama-generated themes"""
    try:
        # Use tinyllama to generate themes
        themes = []
        
        # Process in smaller batches for tinyllama
        for i in range(0, len(problem_texts), 3):  # Even smaller batch
            batch = problem_texts[i:i+3]
            batch_text = "\n".join(f"{j}: {text}" for j, text in enumerate(batch))
            
            prompt = f"""
For each problem, give me 2 words that describe its main issue:

{batch_text}

Format: 1. [word1 word2], 2. [word1 word2], etc.
"""
            
            response = await call_ollama_simple(prompt, model="tinyllama")
            if response:
                # Simple parsing
                theme_lines = response.strip().split(',')
                for line in theme_lines:
                    words = line.strip(' 1234567890.')
                    if words:
                        themes.append(words)
                    else:
                        themes.append("General Issue")
            else:
                themes.extend(["General Issue"] * len(batch))
        
        # Group by similar themes
        clusters_map = defaultdict(list)
        for text, theme in zip(problem_texts, themes):
            clusters_map[theme].append(text)
        
        # Generate solutions
        clusters_data = []
        for theme, problems in clusters_map.items():
            # Simple solution based on theme
            if "financial" in theme.lower() or "money" in theme.lower():
                solution = "Consider budgeting, seeking financial advice, or exploring income opportunities."
            elif "abuse" in theme.lower() or "violence" in theme.lower() or "hits" in theme:
                solution = "Please seek help from authorities or support services immediately. Your safety is important."
            else:
                solution = "Address the underlying issue systematically and seek appropriate help."
            
            clusters_data.append({
                "category": theme.title(),
                "representative_text": problems[0],
                "solution": solution,
                "count": len(problems),
                "problems": problems
            })
        
        logger.info(f"Created {len(clusters_data)} clusters with tinyllama")
        return clusters_data
        
    except Exception as e:
        logger.error(f"Semantic fallback failed: {e}")
        return simple_grouping_fallback(problem_texts)

# Also fix the main ai_cluster_problems function to not default to llama2
async def ai_cluster_problems(problems: List[Problem]) -> List[dict]:
    """Smarter AI-powered clustering using tinyllama"""
    if not problems:
        return []

    logger.info(f"Starting AI clustering on {len(problems)} problems")
    
    problem_texts = [p.text for p in problems]
    
    # Try enhanced clustering with tinyllama
    clusters_data = await intelligent_batch_clustering(problem_texts)
    
    # Generate solutions for each cluster
    for cluster in clusters_data:
        if not cluster["solution"]:
            solution = await generate_simple_solution(cluster["category"], cluster["problems"])
            cluster["solution"] = solution
    
    return clusters_data

async def generate_simple_solution(category: str, problems: List[str]) -> str:
    """Generate solution using tinyllama"""
    sample = problems[0] if len(problems) > 0 else ""
    
    prompt = f"""
Problem category: {category}
Example problem: {sample}

Give a short, helpful solution (1-2 sentences):
"""
    
    response = await call_ollama_simple(prompt, model="tinyllama")
    return response.strip() if response else "Address this issue with appropriate measures."

async def semantic_fallback_clustering(problem_texts: List[str]) -> List[dict]:
    """Enhanced fallback using AI-generated themes instead of hardcoded keywords"""
    try:
        # Use AI to generate themes for each problem
        themes = await generate_problem_themes(problem_texts)
        
        # Cluster based on similar themes
        clusters_map = defaultdict(list)
        theme_to_category = {}
        
        for i, (text, theme) in enumerate(zip(problem_texts, themes)):
            # Find existing similar theme or create new
            matched = False
            for existing_theme, category in theme_to_category.items():
                if are_themes_similar(theme, existing_theme):
                    clusters_map[category].append(text)
                    matched = True
                    break
            
            if not matched:
                # Create new category based on theme
                category_name = theme_to_category_name(theme)
                theme_to_category[theme] = category_name
                clusters_map[category_name].append(text)
        
        # Generate solutions for each cluster
        clusters_data = []
        for category, problems in clusters_map.items():
            solution = await generate_cluster_solution(category, problems)
            
            clusters_data.append({
                "category": category,
                "representative_text": problems[0],
                "solution": solution,
                "count": len(problems),
                "problems": problems
            })
        
        # Merge small clusters
        if len(clusters_data) > 8:
            clusters_data = merge_similar_clusters(clusters_data)
        
        logger.info(f"Semantic fallback created {len(clusters_data)} clusters")
        return clusters_data
        
    except Exception as e:
        logger.error(f"Semantic fallback failed: {e}")
        # Ultimate fallback - simple grouping
        return simple_grouping_fallback(problem_texts)

async def generate_problem_themes(problem_texts: List[str]) -> List[str]:
    """Generate concise themes for each problem using AI"""
    themes = []
    batch_size = 5
    
    for i in range(0, len(problem_texts), batch_size):
        batch = problem_texts[i:i+batch_size]
        batch_text = "\n".join(f"{j}: {text}" for j, text in enumerate(batch))
        
        prompt = f"""
For each problem below, generate a 2-3 word theme that captures its ESSENCE.
Focus on the core issue, not specific details.

Problems:
{batch_text}

Return as JSON: {{"themes": ["theme1", "theme2", ...]}}
"""
        
        response = await call_ollama_simple(prompt)
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > 0:
                    result = json.loads(response[json_start:json_end])
                    themes.extend(result.get("themes", ["General"] * len(batch)))
                else:
                    themes.extend(["General"] * len(batch))
            except:
                themes.extend(["General"] * len(batch))
        else:
            themes.extend(["General"] * len(batch))
    
    return themes

def are_themes_similar(theme1: str, theme2: str, threshold: float = 0.7) -> bool:
    """Determine if two themes are semantically similar"""
    # Simple word overlap (can be enhanced with embeddings)
    words1 = set(theme1.lower().split())
    words2 = set(theme2.lower().split())
    
    if not words1 or not words2:
        return False
    
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return (intersection / union) > threshold if union > 0 else False

def theme_to_category_name(theme: str) -> str:
    """Convert a theme to a readable category name"""
    # Capitalize and clean up
    words = theme.strip().split()
    if len(words) <= 3:
        return " ".join(word.capitalize() for word in words)
    else:
        # Take first 2-3 meaningful words
        meaningful = [w for w in words if len(w) > 2][:3]
        return " ".join(meaningful).title()

async def generate_cluster_solution(category: str, problems: List[str]) -> str:
    """Generate solution for a cluster of problems"""
    sample_problems = problems[:3]  # Use first 3 as examples
    problems_text = "\n".join(f"- {p}" for p in sample_problems)
    
    prompt = f"""
Category: {category}

Problems in this category:
{problems_text}

Generate a concise, actionable solution that addresses the ROOT CAUSE of these related problems.
Focus on practical steps, not general advice.

Solution (2-3 sentences max):
"""
    
    response = await call_ollama_simple(prompt)
    return response.strip() if response else "Investigate these issues systematically and address common patterns."

def merge_similar_clusters(clusters_data: List[dict]) -> List[dict]:
    """Merge clusters with similar categories"""
    merged = {}
    
    for cluster in clusters_data:
        category = cluster["category"]
        # Check for similar existing category
        matched = False
        for existing_cat in merged.keys():
            if are_themes_similar(category, existing_cat):
                # Merge with existing cluster
                merged[existing_cat]["problems"].extend(cluster["problems"])
                merged[existing_cat]["count"] += cluster["count"]
                matched = True
                break
        
        if not matched:
            merged[category] = cluster.copy()
    
    return list(merged.values())

def simple_grouping_fallback(problem_texts: List[str]) -> List[dict]:
    """Final fallback - simple length-based grouping"""
    logger.info("Using simple grouping fallback")
    
    # Group by problem length (simple but often correlates with complexity)
    short = []
    medium = []
    long = []
    
    for text in problem_texts:
        if len(text) < 50:
            short.append(text)
        elif len(text) < 150:
            medium.append(text)
        else:
            long.append(text)
    
    clusters_data = []
    for category, problems in [("Brief Issues", short), 
                               ("Detailed Problems", medium), 
                               ("Complex Descriptions", long)]:
        if problems:
            clusters_data.append({
                "category": category,
                "representative_text": problems[0],
                "solution": "Review these issues based on their complexity level. Prioritize complex descriptions as they often contain more context.",
                "count": len(problems),
                "problems": problems
            })
    
    return clusters_data


app = FastAPI(title="AI Problem Aggregator")
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

MAIN_PAGE_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Problem Aggregator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 600px; margin: 50px auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        h1 { color: #333; margin-bottom: 10px; font-size: 32px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 16px; }
        .alert { padding: 15px; border-radius: 10px; margin-bottom: 20px; font-weight: 500; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        textarea { width: 100%; padding: 15px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 16px; font-family: inherit; resize: vertical; min-height: 120px; transition: border-color 0.3s; }
        textarea:focus { outline: none; border-color: #667eea; }
        .char-count { text-align: right; color: #999; font-size: 14px; margin-top: 5px; margin-bottom: 15px; }
        button { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; font-size: 18px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4); }
        button:active { transform: translateY(0); }
        .admin-link { text-align: center; margin-top: 20px; }
        .admin-link a { color: #667eea; text-decoration: none; font-weight: 500; }
        .admin-link a:hover { text-decoration: underline; }
    </style>
    <script>
        function updateCharCount() {
            const textarea = document.getElementById('problem_text');
            const count = document.getElementById('char-count');
            const length = textarea.value.length;
            count.textContent = length + ' / 200 characters';
            count.style.color = length > 200 ? '#dc3545' : '#999';
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>🤝 Problem Aggregator</h1>
        <p class="subtitle">Submit your problem and we'll help find AI-powered solutions (using TinyLlama)</p>
        
        {% if success %}
        <div class="alert success">✓ Problem submitted successfully! Thank you for sharing.</div>
        {% endif %}
        
        {% if error %}
        <div class="alert error">✗ {{ error }}</div>
        {% endif %}
        
        <form method="post">
            <textarea 
                name="problem_text" 
                id="problem_text" 
                placeholder="Describe your problem here..." 
                required 
                maxlength="200"
                oninput="updateCharCount()"
            ></textarea>
            <div class="char-count" id="char-count">0 / 200 characters</div>
            <button type="submit">Submit Problem</button>
        </form>
        
        <div class="admin-link">
            <a href="/admin">Admin Dashboard</a>
        </div>
    </div>
</body>
</html>'''

ADMIN_PAGE_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Admin Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; }
        .header h1 { font-size: 32px; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 16px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .stat-card h3 { color: #666; font-size: 14px; font-weight: 500; margin-bottom: 10px; text-transform: uppercase; }
        .stat-card .number { font-size: 36px; font-weight: 700; color: #667eea; }
        .alert { padding: 15px 20px; border-radius: 10px; margin-bottom: 20px; font-weight: 500; }
        .alert.success { background: #d4edda; color: #155724; }
        .alert.warning { background: #fff3cd; color: #856404; }
        .alert.error { background: #f8d7da; color: #721c24; }
        .controls { display: flex; gap: 15px; margin-bottom: 30px; flex-wrap: wrap; }
        .btn { padding: 12px 24px; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block; transition: all 0.2s; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-secondary { background: white; color: #667eea; border: 2px solid #667eea; }
        .btn-secondary:hover { background: #667eea; color: white; }
        .clusters-section { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .clusters-section h2 { margin-bottom: 10px; color: #333; }
        .clusters-section p { color: #666; margin-bottom: 25px; }
        .cluster { background: #f8f9fa; padding: 25px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #667eea; }
        .cluster-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px; }
        .cluster-title { font-size: 20px; font-weight: 600; color: #333; }
        .cluster-badge { background: #667eea; color: white; padding: 5px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; }
        .cluster-category { color: #667eea; font-weight: 600; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }
        .cluster-text { color: #555; margin-bottom: 15px; font-style: italic; line-height: 1.6; }
        .solution-box { background: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 2px solid #e0e0e0; }
        .solution-label { font-weight: 600; color: #667eea; margin-bottom: 8px; display: block; }
        .solution-text { color: #333; line-height: 1.6; }
        .problems-toggle { cursor: pointer; color: #667eea; font-weight: 600; margin-top: 10px; user-select: none; }
        .problems-toggle:hover { text-decoration: underline; }
        .problems-list { display: none; margin-top: 15px; padding: 15px; background: white; border-radius: 8px; }
        .problems-list.show { display: block; }
        .problem-item { padding: 10px; border-left: 3px solid #e0e0e0; margin-bottom: 10px; color: #555; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state h3 { font-size: 24px; margin-bottom: 10px; }
    </style>
    <script>
        function toggleProblems(id) {
            const list = document.getElementById('problems-' + id);
            list.classList.toggle('show');
        }
    </script>
</head>
<body>
    <div class="header">
        <h1>🔧 Admin Dashboard</h1>
        <p>AI-powered problem clustering with TinyLlama (free & local)</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Total Problems</h3>
            <div class="number">{{ total_problems }}</div>
        </div>
        <div class="stat-card">
            <h3>Problem Clusters</h3>
            <div class="number">{{ total_clusters }}</div>
        </div>
    </div>
    
    {% if message %}
    <div class="alert {{ message_type }}">{{ message }}</div>
    {% endif %}
    
    <div class="controls">
        <form method="post" action="/admin/cluster" style="display: inline;">
            <button type="submit" class="btn btn-primary">🤖 Run AI Clustering (TinyLlama)</button>
        </form>
        <a href="/" class="btn btn-secondary">← Back to Submission</a>
        <a href="/admin" class="btn btn-secondary">🔄 Refresh</a>
    </div>
    
    {% if clusters %}
    <div class="clusters-section">
        <h2>Problem Clusters & AI Solutions</h2>
        <p>Clustered and analyzed by TinyLlama (local AI)</p>
        
        {% for cluster in clusters %}
        <div class="cluster">
            <div class="cluster-header">
                <div>
                    <div class="cluster-category">{{ cluster.category }}</div>
                    <div class="cluster-title">Cluster {{ loop.index }}</div>
                </div>
                <div class="cluster-badge">{{ cluster.count }} reports</div>
            </div>
            
            <div class="cluster-text">
                <strong>Representative Problem:</strong> {{ cluster.representative_text }}
            </div>
            
            <div class="solution-box">
                <span class="solution-label">💡 AI-Generated Solution:</span>
                <div class="solution-text">{{ cluster.solution }}</div>
            </div>
            
            <div class="problems-toggle" onclick="toggleProblems({{ cluster.id }})">
                ▼ View all {{ cluster.count }} similar problems
            </div>
            <div class="problems-list" id="problems-{{ cluster.id }}">
                {% for problem in cluster.problems %}
                <div class="problem-item">{{ problem }}</div>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="clusters-section">
        <div class="empty-state">
            <h3>No clusters available</h3>
            <p>Click "Run AI Clustering" to analyze submitted problems and generate solutions.</p>
        </div>
    </div>
    {% endif %}
</body>
</html>'''

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request, success: bool = False):
    from jinja2 import Template
    template = Template(MAIN_PAGE_HTML)
    html_content = template.render(success=success)
    return HTMLResponse(content=html_content)

@app.post("/", response_class=HTMLResponse)
async def submit_problem(request: Request, problem_text: str = Form(...)):
    from jinja2 import Template
    
    error_message = None
    if len(problem_text.strip()) < 10:
        error_message = "Problem description is too short. Please provide at least 10 characters."
    elif len(problem_text) > 200:
        error_message = "Problem description is too long. Maximum 200 characters allowed."
    
    if error_message:
        template = Template(MAIN_PAGE_HTML)
        html_content = template.render(error=error_message)
        return HTMLResponse(content=html_content)
    
    await store_problem(problem_text.strip())
    
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
    
    from jinja2 import Template
    template = Template(ADMIN_PAGE_HTML)
    
    if len(problems) < 2:
        clusters = await get_clusters()
        html_content = template.render(
            total_problems=len(problems),
            total_clusters=len(clusters),
            clusters=clusters,
            message="Need at least 2 problems to perform clustering",
            message_type="warning"
        )
        return HTMLResponse(content=html_content)
    
    clusters_data = await ai_cluster_problems(problems)
    
    if not clusters_data:
        clusters = await get_clusters()
        html_content = template.render(
            total_problems=len(problems),
            total_clusters=len(clusters),
            clusters=clusters,
            message="AI clustering failed. Make sure Ollama is running with: ollama serve",
            message_type="error"
        )
        return HTMLResponse(content=html_content)
    
    await store_clusters(clusters_data)
    clusters = await get_clusters()
    
    html_content = template.render(
        total_problems=len(problems),
        total_clusters=len(clusters),
        clusters=clusters,
        message=f"AI clustering completed! Generated {len(clusters_data)} clusters with personalized solutions.",
        message_type="success"
    )
    return HTMLResponse(content=html_content)

@app.get("/api/problems")
async def get_problems():
    problems = await get_all_problems()
    return [{"id": p.id, "text": p.text, "created_at": p.created_at} for p in problems]

@app.get("/api/clusters")
async def get_clusters_api():
    clusters = await get_clusters()
    return [{"id": c.id, "category": c.category, "representative_text": c.representative_text, 
             "solution": c.solution, "count": c.count} for c in clusters]

@app.on_event("startup")
async def startup_event():
    await init_database()
    logger.info("AI Problem Aggregator with TinyLlama started successfully!")
    logger.info("Make sure to run: ollama pull tinyllama")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
