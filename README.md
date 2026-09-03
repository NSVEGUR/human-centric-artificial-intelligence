# Human-Centric Artificial Intelligence

TUHH course project: Django app bundling four sub projects/tasks related to Human-Centric Artificial Intelligence.

## Group

| Name                      | Matriculation No. |
| ------------------------- | ----------------- |
| Nagasai Vegur             | 672843            |
| Sai Adarsh Varma Chittari | 670175            |
| Roshan Srinivasan         | 672856            |

## Setup

Requires Python 3.12+. Pick whichever method you have available.

### Option A — uv

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Then:

```bash
uv sync
uv run python manage.py migrate
uv run python manage.py runserver
```

To activate the venv directly instead of prefixing with `uv run`:
- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

### Option B — pip

```bash
python -m venv .venv

# activate
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows

pip install -e .
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

---

## Tests

```bash
uv run python manage.py test   # uv
python manage.py test          # pip
```

## Project reports

Project 3 and Project 4 each have a "Download PDF Report" button on their landing page,
generated on the fly from `project3/report.py` / `project4/report.py`.