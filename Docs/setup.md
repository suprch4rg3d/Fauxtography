# Setup & Environment

This project can be installed two ways — with **Poetry** or with **plain pip**
(via `requirements.txt`). Use whichever you're comfortable with; both produce
the same runtime environment. Pick one per machine and stick to it.

**Prerequisite:** Python **3.14** (3.14 is required; the dependency set is
verified against macOS/arm64 + Python 3.14 + torch 2.15).

---

## Option A — Poetry (preferred by the repo author)

Poetry 2.x reads the PEP 621 fields in `pyproject.toml` directly. Versions come
from `poetry.lock` once generated, so teammates using Poetry get identical
environments automatically.

### 1. Install Poetry (if not already installed)

```bash
# macOS / Linux (curl install script)
curl -sSL https://install.python-poetry.org | python3 -

# or via Homebrew
brew install poetry
```

Verify: `poetry --version` (created with 2.3+).

### 2. Create the environment and install dependencies

```bash
cd fauxtography
poetry install --all-extras
```

- `--all-extras` installs the runtime deps **and** the dev extras
  (`pytest`, `ruff`). If you only want runtime deps: `poetry install`.
- `virtualenvs.in-project` is enabled, so the venv is created at
  `.venv/` inside the repo.
- This generates `poetry.lock` on first run — **commit it** so your teammate
  gets the same pinned versions.

### 3. Configure the interpreter in your IDE

Point your editor at the Poetry-managed venv:

```bash
poetry env info --path      # prints the venv path
```

- **VS Code:** select the interpreter shown above.
- **PyCharm:** Settings → Project → Python Interpreter → Add → Existing,
  browse to the path.

### 4. Everyday commands

```bash
poetry run python scripts/prepare_data.py --help   # run a script (auto-venv)
poetry run pytest                                   # run tests in the venv
poetry add <package>                                # add a runtime dep
poetry add --group dev <package>                    # add a dev dep
poetry run python -m pip list                       # inspect installed
```

Use `poetry shell` first (leaves you inside the venv) if you prefer typing
plain `python` / `pytest`.

### 5. Keeping the lockfile in sync

If `pyproject.toml` changes (a teammate adds a dependency):

```bash
poetry lock        # re-resolve and update poetry.lock
poetry install     # sync your venv with the lockfile
```

---

## Option B — Plain pip (requirements.txt)

If you don't use Poetry, install exactly the same pinned versions from the
requirements files with the standard `venv` + `pip` flow. Works on any setup.

### 1. Create and activate a virtual environment

```bash
cd fauxtography
python3.14 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt          # runtime deps (pinned)
pip install -r requirements-dev.txt      # + dev tooling (pytest, ruff)
```

or, if dev tooling isn't needed yet, just:

```bash
pip install -r requirements.txt
```

### 3. Install the project itself (editable)

The scripts and tests import the `fauxtography` package, so install it in
editable mode:

```bash
pip install -e .
```

### 4. Everyday commands

```bash
python scripts/prepare_data.py --help   # run a script (venv active)
pytest                                  # run tests
pip install <package>                   # add a dep
pip freeze > requirements.txt           # regenerate pins (see note below)
```

---

## Keeping the two install methods in agreement

`requirements.txt` is the **verified, pinned** ground truth for plain pip
users. Two simple rules keep both paths consistent:

1. When adding a dependency, add it to **both** `pyproject.toml`
   (`[project.dependencies]` / `[project.optional-dependencies].dev`) and the
   matching requirements file, using the same version constraint.
2. Re-verify the environment after any change and, if you're the one who
   installs via pip, regenerate the pins:

```bash
pip freeze | grep -iE "^(torch|torchvision|transformers|numpy|pillow|pyyaml|scikit-learn|matplotlib|pandas|tqdm)="
```

Update `requirements.txt` accordingly. (Poetry users get pins automatically
from `poetry.lock`.)

---

## Troubleshooting

- **``ModuleNotFoundError: fauxtography``** — the package isn't installed
  editable. Run `pip install -e .` (pip) or `poetry install` (Poetry).
- **Wrong Python version** — the project needs 3.14. Check `python3.14 --version`;
  if 3.14 isn't on your PATH, install it (e.g. `brew install python@3.14`).
- **`pytest` not found** — dev extras were skipped. Use `--all-extras`
  (Poetry) or `requirements-dev.txt` (pip).
- **Ideas for GPU vs CPU** — torch 2.15 wheels are the same across CUDA/CPU
  installs on Apple Silicon; on a machine with an NVIDIA GPU the CUDA wheels
  install automatically via pip. No extra steps needed for development.