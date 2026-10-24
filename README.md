# Mini AI Business Intelligence Assistant

> **Status:** Work in progress — README will be completed in Step 10.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd AI_assistant

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up secrets
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run ETL (loads CSV → SQLite)
python -m app.etl

# 6. Start the server
uvicorn app.main:app --reload

# 7. Open http://localhost:8000
```

## Run Tests

```bash
pytest tests/ -v
```

## Architecture

_Diagram and full explanation coming in Step 10._
