# Human-Centric Artificial Intelligence

TUHH course project: Django app bundling four sub projects/tasks related to Human-Centric Artificial Intelligence.

## Group

| Name                      | Matriculation No. |
| ------------------------- | ----------------- |
| Nagasai Vegur             | 672843            |
| Sai Adarsh Varma Chittari | 670175            |
| Roshan Srinivasan         | 672856            |

## Setup (new machine)

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
This project is developed and tested with **Python 3.12.10** and **uv 0.9.15**, so if something breaks
on a newer/older sub version, pin to these first.

```bash
# 1. Install uv, if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. Clone and enter the project root (the folder containing manage.py)
git clone <url> <folder_name>
cd <folder_name>

# 3. Create the venv + install dependencies, then run
uv sync
uv run python manage.py migrate
uv run python manage.py runserver
```

`uv sync` creates a local `.venv/` and installs everything into it. And no separate
`python -m venv` step needed. To activate it directly instead of prefixing commands
with `uv run`: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate`
(Windows).

Open http://127.0.0.1:8000/

## Tests

```bash
uv run python manage.py test
```

## Project reports

Project 3 and Project 4 each have a "Download PDF Report" button on their landing page,
generated on the fly from `project3/report.py` / `project4/report.py`.
