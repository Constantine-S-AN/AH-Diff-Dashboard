"""Print derived GitHub Pages links from git remote origin."""

from __future__ import annotations

import re
import subprocess
import sys


def _get_origin_url() -> str | None:
    """Return remote origin URL, or `None` if unavailable."""

    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _parse_owner_repo(remote_url: str) -> tuple[str, str] | None:
    """Parse owner/repo from GitHub https or ssh remotes."""

    patterns = [
        r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, remote_url.strip())
        if not match:
            continue
        owner = match.group("owner")
        repo = match.group("repo")
        return owner, repo
    return None


def _build_pages_url(owner: str, repo: str) -> str:
    """Build Pages URL from owner and repo."""

    owner_norm = owner.lower()
    repo_norm = repo
    if repo_norm.lower() == f"{owner_norm}.github.io":
        return f"https://{owner_norm}.github.io/"
    return f"https://{owner_norm}.github.io/{repo_norm}/"


def main() -> None:
    """Print pages_url, demo_index, and demo_report."""

    placeholder = "https://<OWNER>.github.io/<REPO>/"
    remote_url = _get_origin_url()
    parsed = _parse_owner_repo(remote_url) if remote_url else None

    if parsed is None:
        pages_url = placeholder
        demo_index = pages_url
        demo_report = f"{pages_url}reports/cost_sensitivity_demo.html"
        print(f"pages_url={pages_url}")
        print(f"demo_index={demo_index}")
        print(f"demo_report={demo_report}")
        print("hint=运行 scripts/print_links.py 生成链接", file=sys.stderr)
        return

    owner, repo = parsed
    pages_url = _build_pages_url(owner, repo)
    demo_index = pages_url
    demo_report = f"{pages_url}reports/cost_sensitivity_demo.html"
    print(f"pages_url={pages_url}")
    print(f"demo_index={demo_index}")
    print(f"demo_report={demo_report}")


if __name__ == "__main__":
    main()
