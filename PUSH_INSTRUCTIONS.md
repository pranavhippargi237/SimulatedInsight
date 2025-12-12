# Pushing to GitHub

## Quick Push Command

```bash
git push -u origin main
```

## If Authentication is Required

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. When prompted for password, use the token instead

### Option 2: SSH Key
1. Set up SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys
3. Change remote URL: `git remote set-url origin git@github.com:pranavhippargi237/SimulatedInsight.git`

## What's Being Pushed

- Full backend (FastAPI, LangChain, ML models)
- Frontend (React, TailwindCSS)
- Docker configuration
- All analysis modules (detection, optimization, simulation, etc.)
- Documentation

## Excluded (via .gitignore)

- `venv/` and `backend/venv/`
- `__pycache__/`
- `.env` files (API keys)
- Database files (`*.db`, `*.sqlite`)
- Node modules
- Build artifacts

## After Pushing

Your repository will be live at:
https://github.com/pranavhippargi237/SimulatedInsight
