# Problem Aggregator

A lightweight web application that collects user-submitted problems, clusters similar issues using machine learning, and suggests automated solutions.

## Features

- **Problem Submission**: Simple web form for users to submit problems (max 200 characters)
- **Intelligent Clustering**: Uses TF-IDF vectorization and MiniBatch K-Means to group similar problems
- **Automated Solutions**: Rule-based solution generation based on problem keywords
- **Admin Dashboard**: View all problems, clusters, and generated solutions
- **RESTful API**: Access problems and clusters programmatically

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL
- **ML**: scikit-learn (TF-IDF, K-Means clustering)
- **Frontend**: Bootstrap 5, Jinja2 templates

## Prerequisites

- Python 3.8+
- PostgreSQL 12+

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd problem-aggregator
```

### 2. Set Up PostgreSQL Database

Install PostgreSQL if you haven't already, then create a database:

```bash
# Start PostgreSQL service (Linux/Mac)
sudo service postgresql start

# Or on Mac with Homebrew
brew services start postgresql

# Create database
psql -U postgres
CREATE DATABASE problem_aggregator;
\q
```

**Update Database Credentials**: Edit the `DATABASE_URL` in `problem_aggregator.py`:

```python
DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost/problem_aggregator"
```

Replace `YOUR_PASSWORD` with your PostgreSQL password.

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Start the Server

```bash
python problem_aggregator.py
```

The application will start at `http://127.0.0.1:8000`

### Access the Application

- **Public Submission Form**: http://127.0.0.1:8000
- **Admin Dashboard**: http://127.0.0.1:8000/admin
  - Username: (leave blank or enter anything)
  - Password: `maramdaas`

## Usage Guide

### Submitting Problems

1. Navigate to http://127.0.0.1:8000
2. Enter a problem description (10-200 characters)
3. Click "Submit Problem"

### Admin Dashboard

1. Go to http://127.0.0.1:8000/admin
2. Log in with password: `maramdaas`
3. Click "Run Clustering" to analyze submitted problems
4. View clusters with suggested solutions
5. Expand clusters to see all similar problems

### API Endpoints

**Get All Problems**
```bash
curl http://127.0.0.1:8000/api/problems
```

**Get All Clusters**
```bash
curl http://127.0.0.1:8000/api/clusters
```

## How It Works

1. **Problem Collection**: Users submit problems via web form
2. **Keyword Extraction**: System extracts relevant keywords using frequency analysis
3. **Clustering**: TF-IDF vectorization converts text to numerical features, MiniBatch K-Means groups similar problems
4. **Solution Generation**: Rule-based system suggests solutions based on problem keywords
5. **Display**: Admin dashboard shows clusters with representative problems and solutions

## Security Notes

⚠️ **Important**: Change the admin password in production!

Edit `ADMIN_PASSWORD` in `problem_aggregator.py`:
```python
ADMIN_PASSWORD = "your_secure_password_here"
```

## Configuration

### Adjust Clustering Parameters

In `problem_aggregator.py`, modify the `perform_lightweight_clustering` function:

```python
def perform_lightweight_clustering(problems: List[Problem], max_clusters: int = 15):
    # Change max_clusters to adjust maximum number of problem groups
```

### Modify Solution Templates

Edit the `solutions` dictionary in `generate_simple_solution()`:

```python
solutions = {
    'money': "Your custom financial advice...",
    'work': "Your custom career advice...",
    # Add more categories
}
```

## Troubleshooting

### Database Connection Error

```
asyncpg.exceptions.InvalidPasswordError
```

**Solution**: Update the `DATABASE_URL` with correct PostgreSQL credentials.

### Port Already in Use

```
ERROR: [Errno 48] Address already in use
```

**Solution**: Change the port in the last line of `problem_aggregator.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8001)  # Changed from 8000
```

### Import Errors

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Project Structure

```
problem-aggregator/
├── problem_aggregator.py    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── venv/                     # Virtual environment (created during setup)
```

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Future Enhancements

- User authentication system
- Export clusters as CSV/PDF
- Integration with GPT for better solution generation
- Real-time clustering updates
- Email notifications for cluster matches
- Multi-language support
