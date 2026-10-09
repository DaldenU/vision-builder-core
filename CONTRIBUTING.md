# Contributing to Vision Builder Core

Thank you for investing your time in contributing to **Vision Builder Core**!

This project adheres to scientific software engineering standards, continuous integration practices, and test-driven development (TDD).

---

## 1. Code of Conduct

Please maintain professional, respectful, and constructive communication at all times.

---

## 2. Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DaldenU/vision-builder-core.git
   cd vision-builder-core
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # Or on Windows:
   .venv\Scripts\activate
   ```

3. **Install dependencies in development mode:**
   ```bash
   pip install --upgrade pip
   pip install -e .[dev]
   ```

---

## 3. Quality Standards & Verification

Before opening a pull request, verify all local quality gates pass:

```bash
# 1. Format code with Black
black vision_builder tests

# 2. Lint with Flake8
flake8 vision_builder tests --max-line-length=88 --extend-ignore=E203,W503

# 3. Static type check with Mypy
mypy vision_builder tests

# 4. Security vulnerability scan with Bandit
bandit -r vision_builder -ll

# 5. Execute automated test suite with coverage
pytest --cov=vision_builder --cov-report=term-missing
```

Coverage must remain **above 90%** for all merged code.

---

## 4. Git Branching & Commit Conventions

- Work on feature branches branched from `main`:
  `git checkout -b feature/your-feature-name`
- Follow **Conventional Commits**:
  - `feat: add optical flow motion filter`
  - `fix: correct bounding box iou edge case`
  - `test: expand telemetry sliding window coverage`
  - `docs: update quickstart guide`
  - `ci: add python 3.12 matrix target`

---

## 5. Pull Request Process

1. Ensure all CI workflow checks pass in GitHub Actions.
2. Provide a clear description of changes, motivation, and test evidence.
3. Obtain at least one review approval before merging into `main`.
