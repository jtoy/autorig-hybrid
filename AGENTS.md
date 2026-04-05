# AGENTS.md - Developer & AI Guide

This document serves as the primary technical reference for AI agents (like yourself) and developers working on the repository. 
Follow these guidelines to maintain consistency and safety.

---

## 🚀 CRITICAL WORKFLOW: GitHub Flow & Branching

Coding directly on the `main` branch is strictly prohibited. We use **GitHub Flow**:

- All work happens in feature branches
- Changes are submitted via Pull Requests
- PRs are reviewed and then **squash merged** into main
- Never merge directly to main without PR review
- Branch convention is issue<:number> or <desc> if there is no issue open.

### Branch Verification & Worktree Protocol
Before starting any work, you MUST verify your current environment:

1. **Verify Environment:** Check both the current branch and the working directory path.
2. **Determine if on a Working Branch:**
   - Run `git branch --show-current`.
   - Check if the current absolute path contains the `.worktrees/` directory.
   - **If you are NOT on `main` OR the path contains `.worktrees/`:** You are already in a working branch. DO NOT create a new worktree unless explicitly requested. Proceed with your task in the current directory.
3. **If you ARE on `main` AND the path does NOT contain `.worktrees/`:**
   - **Sync main:** Always ensure your local `main` is up to date: `git pull origin main`.
   - **Create Worktree:** Create a separate workspace for your branch:
     ```bash
      git worktree add ./.worktrees/<branch_name> -b <branch_name>
      ```
    - **Switch Directory:** Change your working directory to the newly created path and perform all operations from there.
4. **Push your work once completed** Once the work is completed we do a final push to the repo.
5. **Pull origin main to include main changes** Once we have done a final push we need to apply current existing changes into our work. We must solve all the conflicts. NEVER use rebase or anything that rewrites the git history.
6. **Never merge yourself** Unless you are explictly told never merge the work. If you are said so use always `squash` merge, atterwards remove the the worktree should be removed to keep the system clean.

---

## 📝 Issues & Documentation Protocol

All project management and documentation activities must follow these rules:

1. **Language:** ALWAYS use English for Issue titles, bodies, and any documentation files.
2. **Issue Creation:**
   - **Title:** Concise and descriptive (e.g., `fix: incorrect status code on missing endpoints`).
   - **Body:** Recommended to include:
     - **Summary:** Brief explanation of the problem or feature.
     - **Context:** Mention specific files, functions, or endpoints involved.
     - **Steps to Reproduce:** (For bugs) Clear steps or expected vs. actual behavior.
   - **Verification:** Always verify the issue was created correctly using `gh issue view`.
3. **Documentation:**
   - Keep it technical, concise, and up to date.
   - Use standard Markdown formatting.

---

## 🛠 Commands & Scripts

### 0. Environment Setup
- **Conda Environment:** Always work in `conjurors` (`conda activate conjurors`).
- **Package Installs:** Do not install global/system packages. Only install inside `conjurors` and only with explicit approval.
- **Node Version:** Node 18.x is required.
- **nvm:** Use `nvm use 18` to ensure the correct version.
- **Package Manager:** `yarn` is mandatory. Do not use `npm` for installing dependencies.

---

## 🎨 Code Style & Standards

### 1. Linting & Formatting
- **Linting:** Use `ruff` for linting; unused variables are errors.
- **Formatting:** Use `black` (or `ruff format`) for consistent formatting.
- **Run Checks:** `ruff check .` and `black .` (or `ruff format .`).

### 2. Naming & Modules
- **Files/Modules:** `snake_case` (e.g., `image_utils.py`).
- **Classes:** `PascalCase`.
- **Functions/Variables:** `snake_case`.
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_CALL_DURATION`).

### 3. Import Organization
Group imports with a single blank line between groups:
1. Python standard library.
2. Third-party packages.
3. Local application modules.

---

## 🔐 Security
- NEVER hardcode secrets.
- Use `python-dotenv` and call `load_dotenv()` at the very top of entry files.
- Ensure `.env` and virtual environments (e.g., `.venv/`, `venv/`) are in `.gitignore`.

---