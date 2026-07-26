# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

# ================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# ================================

"""Check if all files contain the required header."""

from pathlib import Path


ROOT = Path(__file__).parent

# Headers for each file type
HEADERS = {
    ".py": """# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
\n""",
    ".yml": """# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
\n""",
    ".yaml": """# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
\n""",
    ".md": """<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->
\n""",
}

# Files to exclude
EXCLUDE_FILES: set[str] = {
    "PULL_REQUEST_TEMPLATE.md",
    "bug_report.md",
    "feature_request.md",
    "CLAUDE.md",  # local, gitignored Claude Code guidance — not shipped source
}

# Directories to exclude (virtualenvs, build artifacts, caches, generated sites).
EXCLUDE_PATHS: set[str] = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "site",
    "plans",
    ".ray",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "egg-info",
}


def check_header(file_path: Path, header: str):
    """Check if the file contains the required header."""
    with file_path.open("r", encoding="utf-8") as file:
        content = file.read()
        return content.startswith(header) or content == header[:-1]


def main():
    """Check if all files contain the required headers."""
    missing_headers = []
    for file_path in ROOT.rglob("*"):
        if not any(part.name in EXCLUDE_PATHS or part.name.endswith(".egg-info") for part in file_path.parents):
            ext = file_path.suffix
            if ext in HEADERS and file_path.name not in EXCLUDE_FILES:
                if not check_header(file_path, HEADERS[ext]):
                    missing_headers.append(str(file_path))

    if missing_headers:
        exit(Exception("The following files are missing the required headers:\n" + "\n- ".join(missing_headers)))


if __name__ == "__main__":
    main()
