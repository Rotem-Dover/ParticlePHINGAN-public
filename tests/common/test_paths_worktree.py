"""Worktree storage resolution.

`storage/` is gitignored and only ever populated in the main checkout (it is
~16 GB, never duplicated per worktree). When the code runs from a *linked* git
worktree, `paths.STORAGE` must therefore point at the main working tree's
`storage/`, not the worktree's own (empty) one. See `_main_worktree_storage`.
"""
from pathlib import Path

from common.paths import _main_worktree_storage


def _make_main_repo(tmp_path: Path) -> Path:
    """A normal checkout: <root>/.git/ (a dir) + <root>/storage/."""
    root = tmp_path / "ParticlePHINGAN"
    (root / ".git").mkdir(parents=True)
    (root / "storage").mkdir()
    return root


def _make_linked_worktree(main_root: Path, name: str) -> Path:
    """Mirror `git worktree add`: a worktree dir whose `.git` is a *file*
    pointing at <main>/.git/worktrees/<name>, with a `commondir` back-link."""
    wt_admin = main_root / ".git" / "worktrees" / name
    wt_admin.mkdir(parents=True)
    (wt_admin / "commondir").write_text("../..\n")

    wt = main_root / ".claude" / "worktrees" / name
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {wt_admin}\n")
    return wt


def test_linked_worktree_resolves_to_main_storage(tmp_path):
    main_root = _make_main_repo(tmp_path)
    wt = _make_linked_worktree(main_root, "FixMCPDFDisparancies")

    assert _main_worktree_storage(wt) == main_root / "storage"


def test_normal_checkout_returns_none(tmp_path):
    main_root = _make_main_repo(tmp_path)
    # `.git` is a directory in a normal checkout -> not a linked worktree.
    assert _main_worktree_storage(main_root) is None


def test_non_repo_returns_none(tmp_path):
    assert _main_worktree_storage(tmp_path) is None
