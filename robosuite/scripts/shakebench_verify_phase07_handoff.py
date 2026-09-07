"""Verify the executable Phase 07R6.1 Git and publication handoff contract.

The verifier treats package evidence, detached R6 publication metadata, Git
objects, tags, ancestry, and the complete commit ranges as authority.  A
stored ``passed`` boolean or a well-shaped hash is never sufficient by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_FILENAME = "docs/phase_07_r6_1_manifest.json"
R6_MANIFEST_FILENAME = "docs/phase_07_r6_manifest.json"
MAP_FILENAME = "docs/shakebench_history_rewrite_map_v2.json"
RUNTIME_MAP_FILENAME = "robosuite/models/assets/shakebench_history_rewrite_map_v2.json"
INDEX_FILENAME = "docs/shakebench_evidence_index_v1.json"
REMOVED_PATH_FILENAME = "docs/shakebench_evidence_removed_paths_v1.txt"
RELEASE_MANIFEST_FILENAME = "docs/shakebench_evidence_release_manifest_v1.json"
GATE_RELEASE_MANIFEST_FILENAME = "docs/shakebench_gate_release_manifest_v1.json"
GATE_SUMS_FILENAME = "docs/shakebench_gate_SHA256SUMS"
DEV_ASSET = "robosuite/models/assets/shakebench_states_dev.json"

R6_IMPLEMENTATION_COMMIT = "a87a87c3540b85894343ad49e95b6ed598b7f7b5"
R6_HANDOFF_COMMIT = "1824060986949c3dd618075499fb8a709e08fb36"
R6_TAG = "shakebench-evidence-v0-2026-09-r6"
R6_1_TAG = "shakebench-gate-v0-2026-09-r6-1"
R6_MANIFEST_SHA256 = "3da368f3f4d78c19f749b7520326ff0e236e28764a24383b946c49215d588474"
R6_RELEASE_MANIFEST_SHA256 = "0ab68826456e0e930c57b9963a690a3e7a2782767ef344db738eef3f20f2c494"
R6_INDEX_SHA256 = "8fce997866776189951ce5f01b3c35ffc3fa427c369fb1e1b997c0a4b28b26c9"
R6_MAP_SHA256 = "551036b0f8c242450830591761d35dceca17260613c3f585c82836c8544f5493"
R6_REMOVED_PATHS_SHA256 = "acf4a5fbbd08b6fa02264def85115600ac515a31f85c31f893e48a7c0547fe3a"
R6_ARCHIVE_SHA256 = "dda863dd9941a498f0702b88105a7b50bceb523d59b7fc29cd48a2cdc9596553"
R6_ARCHIVE_FILENAME = "shakebench-evidence-v0-2026-09-r6.tar.zst"
R6_ARCHIVE_URL = (
    "https://github.com/Blien2002/ShakeBench/releases/download/"
    "shakebench-evidence-v0-2026-09-r6/shakebench-evidence-v0-2026-09-r6.tar.zst"
)

ALLOWED_R6_HANDOFF_FILES = {
    R6_MANIFEST_FILENAME,
    "docs/phase_07_report.md",
    "docs/shakebench_prompts/README.md",
    INDEX_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    "SHA256SUMS",
}
DEFAULT_R6_1_POST_HANDOFF_FILES: set[str] = set()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _git(repo: Path, *args: str, check: bool = False) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _git_exists(repo: Path, commit: str) -> bool:
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        return False
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _commit_tree(repo: Path, commit: str) -> str | None:
    if not _git_exists(repo, commit):
        return None
    value = _git(repo, "rev-parse", f"{commit}^{{tree}}")
    return value if HEX40.fullmatch(value) else None


def _commit_file_sha256(repo: Path, commit: str, path: str) -> str | None:
    if not _git_exists(repo, commit):
        return None
    completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return _sha256_bytes(completed.stdout)


def _commit_paths(repo: Path, commit: str) -> set[str]:
    output = _git(repo, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return {line for line in output.splitlines() if line}


def _commits_between(repo: Path, ancestor: str, descendant: str) -> list[str]:
    if ancestor == descendant:
        return []
    output = _git(repo, "rev-list", "--reverse", "--ancestry-path", f"{ancestor}..{descendant}")
    return [line for line in output.splitlines() if line]


def _safe_repo_path(repo: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{label} is not a safe repository-relative path")
    path = (repo / value).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    return path


def _verify_commit_range(
    repo: Path,
    start: str,
    end: str,
    allowed_files: Iterable[str],
    label: str,
) -> list[str]:
    errors: list[str] = []
    allowed = set(allowed_files)
    if not _git_exists(repo, start) or not _git_exists(repo, end):
        return [f"{label} range contains a missing commit"]
    if not _ancestor(repo, start, end):
        return [f"{label} start is not an ancestor of end"]
    for commit in _commits_between(repo, start, end):
        changed = _commit_paths(repo, commit)
        unexpected = sorted(changed - allowed)
        if unexpected:
            errors.append(f"{label} commit {commit} changes unauthorized files: {', '.join(unexpected)}")
    return errors


def _verify_remote_tag(repo: Path, remote: str, tag: str, expected_commit: str) -> str | None:
    completed = subprocess.run(
        ["git", "ls-remote", remote, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"cannot query remote tag {tag}: {completed.stderr.strip()}"
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[1]] = fields[0]
    peeled = values.get(f"refs/tags/{tag}^{{}}")
    if peeled != expected_commit:
        return f"remote tag {tag} peeled commit mismatch: {peeled!r} != {expected_commit}"
    if f"refs/tags/{tag}" not in values:
        return f"remote tag {tag} is missing its tag ref"
    return None


def _tag_commit(repo: Path, tag: str) -> str | None:
    value = _git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    return value if HEX40.fullmatch(value) else None


def _verify_tag(
    repo: Path,
    tag: str,
    expected_commit: str,
    current_head: str,
    bound_files: Iterable[str],
    *,
    remote: str | None = None,
) -> list[str]:
    errors: list[str] = []
    actual_commit = _tag_commit(repo, tag)
    if actual_commit is None:
        errors.append(f"Git tag is missing or not a commit-ish: {tag}")
        return errors
    if actual_commit != expected_commit:
        errors.append(f"Git tag {tag} points to {actual_commit}, expected {expected_commit}")
    if not _ancestor(repo, actual_commit, current_head):
        errors.append(f"Git tag {tag} commit is not an ancestor of current HEAD")
    for path in bound_files:
        current = repo / path
        expected = _sha256(current) if current.is_file() else None
        actual = _commit_file_sha256(repo, actual_commit, path)
        if expected is None or actual != expected:
            errors.append(f"Git tag {tag} tree binding mismatch: {path}")
    if remote is not None:
        remote_error = _verify_remote_tag(repo, remote, tag, expected_commit)
        if remote_error:
            errors.append(remote_error)
    return errors


def _verify_map(repo: Path, map_path: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        mapping = _read_json(map_path, "history rewrite map v2")
    except ValueError as exc:
        return {}, [str(exc)]
    if mapping.get("schema_id") != "shakebench.history_rewrite_map" or mapping.get("schema_version") != 2:
        errors.append("history rewrite map schema/version mismatch")
    payload = dict(mapping)
    expected_payload_hash = payload.pop("mapping_payload_sha256", None)
    if not isinstance(expected_payload_hash, str) or not HEX64.fullmatch(expected_payload_hash):
        errors.append("history rewrite map payload hash is malformed")
    elif _sha256_bytes(_canonical(payload)) != expected_payload_hash:
        errors.append("history rewrite map payload hash mismatch")
    history = mapping.get("history_rewrite")
    if not isinstance(history, dict):
        errors.append("history rewrite details are missing")
    else:
        if history.get("target_refs") != ["refs/heads/master"]:
            errors.append("history rewrite target refs are not explicit")
        if history.get("path_list") != REMOVED_PATH_FILENAME:
            errors.append("history rewrite path-list binding mismatch")
        removed = repo / REMOVED_PATH_FILENAME
        if not removed.is_file() or history.get("path_list_sha256") != _sha256(removed):
            errors.append("history rewrite path-list hash mismatch")
        if history.get("removed_path_count") != 153:
            errors.append("history rewrite removed-path count mismatch")
        tool = history.get("filter_tool")
        if not isinstance(tool, dict) or tool.get("name") != "git-filter-repo" or not tool.get("version"):
            errors.append("filter-repo tool/version binding is missing")
    mappings = mapping.get("commit_mapping")
    if not isinstance(mappings, list) or len(mappings) < 3:
        errors.append("old-to-new commit mapping is incomplete")
        mappings = []
    required_roles = {"pre_history_rewrite_anchor", "pre_rewrite_implementation", "slimming_source"}
    seen_roles: set[str] = set()
    for row in mappings:
        if not isinstance(row, dict):
            errors.append("commit mapping row is not an object")
            continue
        role = row.get("role")
        original = row.get("original_commit")
        rewritten = row.get("rewritten_commit")
        if not isinstance(role, str) or role in seen_roles or role not in required_roles:
            errors.append("commit mapping role is missing, unknown, or duplicated")
        else:
            seen_roles.add(role)
        if not isinstance(original, str) or not HEX40.fullmatch(original):
            errors.append(f"invalid original commit for {role}")
        if not isinstance(rewritten, str) or not HEX40.fullmatch(rewritten):
            errors.append(f"invalid rewritten commit for {role}")
        elif not _git_exists(repo, rewritten):
            errors.append(f"rewritten commit does not exist: {rewritten}")
    if seen_roles != required_roles:
        errors.append("old-to-new commit mapping roles are incomplete")
    anchor = mapping.get("rewritten_anchor")
    if not isinstance(anchor, dict):
        errors.append("rewritten dev-state anchor is missing")
        anchor = {}
    rewritten_anchor = anchor.get("rewritten_commit")
    old_anchor = anchor.get("pre_history_rewrite_commit")
    if not isinstance(rewritten_anchor, str) or not HEX40.fullmatch(rewritten_anchor):
        errors.append("rewritten anchor commit is malformed")
    elif not _git_exists(repo, rewritten_anchor):
        errors.append("rewritten anchor commit does not exist")
    if not isinstance(old_anchor, str) or not HEX40.fullmatch(old_anchor):
        errors.append("pre-rewrite anchor commit is malformed")
    else:
        current_head = _git(repo, "rev-parse", "HEAD")
        if _ancestor(repo, old_anchor, current_head):
            errors.append("pre-rewrite anchor is still reachable from current HEAD")
    if anchor.get("asset_path") != DEV_ASSET:
        errors.append("rewritten anchor asset path mismatch")
    protected = manifest.get("protected_science")
    if anchor.get("asset_sha256") != (protected or {}).get("dev_state_asset_sha256"):
        errors.append("manifest/map dev-state asset hash mismatch")
    if not isinstance(anchor.get("asset_sha256"), str) or not HEX64.fullmatch(anchor["asset_sha256"]):
        errors.append("rewritten anchor asset hash is malformed")
    else:
        rewritten_asset_hash = _commit_file_sha256(repo, str(rewritten_anchor), DEV_ASSET)
        current_asset = repo / DEV_ASSET
        if rewritten_asset_hash != anchor["asset_sha256"]:
            errors.append("rewritten anchor dev-state asset content hash mismatch")
        if not current_asset.is_file() or _sha256(current_asset) != anchor["asset_sha256"]:
            errors.append("current dev-state asset content hash mismatch")
    expected_tree = anchor.get("tree_sha1")
    if not isinstance(expected_tree, str) or not HEX40.fullmatch(expected_tree):
        errors.append("rewritten anchor tree hash is malformed")
    elif _commit_tree(repo, str(rewritten_anchor)) != expected_tree:
        errors.append("rewritten anchor tree hash mismatch")
    for key, commit in (mapping.get("implementation_commits") or {}).items():
        if key in {"pre_rewrite", "slimming_source"}:
            continue
        if not isinstance(commit, str) or not HEX40.fullmatch(commit) or not _git_exists(repo, commit):
            errors.append(f"implementation commit does not exist: {key}")
    tree_bindings = mapping.get("relevant_tree_bindings")
    if not isinstance(tree_bindings, dict):
        errors.append("relevant tree bindings are missing")
    else:
        for field, commit_key in (
            ("rewritten_anchor_tree_sha1", "rewritten_anchor"),
            ("rewritten_implementation_tree_sha1", "rewritten_implementation"),
            ("r5_handoff_tree_sha1", "r5_handoff"),
        ):
            commit = (mapping.get("implementation_commits") or {}).get(commit_key)
            if isinstance(commit, str) and _commit_tree(repo, commit) != tree_bindings.get(field):
                errors.append(f"relevant tree binding mismatch: {field}")
    science = mapping.get("science_bindings")
    protected = manifest.get("protected_science")
    if not isinstance(science, dict) or not isinstance(protected, dict):
        errors.append("science bindings are missing")
    elif any(science.get(key) != protected.get(key) for key in science):
        errors.append("science binding changed across map and manifest")
    return mapping, errors


def _package_fields(package: Mapping[str, Any], kind: str) -> tuple[Any, Any, Any]:
    prefix = f"{kind}_"
    return package.get(prefix + "filename"), package.get(prefix + "bytes"), package.get(prefix + "sha256")


def validate_package_evidence(payload: Mapping[str, Any], package: Mapping[str, Any]) -> list[str]:
    """Validate the complete package authority payload against manifest fields."""

    errors: list[str] = []
    if not isinstance(payload, Mapping) or not isinstance(package, Mapping):
        return ["package evidence and manifest binding must be objects"]
    if payload.get("schema_id") != "shakebench.phase07.package_evidence" or payload.get("schema_version") != 2:
        errors.append("package evidence schema must be v2")
    if payload.get("passed") is not True:
        errors.append("package evidence top-level passed is not true")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2:
        errors.append("package evidence must contain exactly two records")
        records = records if isinstance(records, list) else []
    by_kind: dict[str, list[Mapping[str, Any]]] = {"wheel": [], "sdist": []}
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("package evidence record is not an object")
            continue
        kind = record.get("kind")
        if kind not in by_kind:
            errors.append(f"unexpected package evidence record kind: {kind!r}")
            continue
        by_kind[kind].append(record)
    for kind in ("wheel", "sdist"):
        rows = by_kind[kind]
        if len(rows) != 1:
            errors.append(f"package evidence must contain exactly one {kind} record")
            continue
        record = rows[0]
        expected_filename, expected_bytes, expected_sha = _package_fields(package, kind)
        if record.get("artifact_filename") != expected_filename:
            errors.append(f"package {kind} filename binding mismatch")
        if record.get("artifact_bytes") != expected_bytes:
            errors.append(f"package {kind} byte-size binding mismatch")
        if (
            record.get("artifact_sha256") != expected_sha
            or not isinstance(expected_sha, str)
            or not HEX64.fullmatch(expected_sha)
        ):
            errors.append(f"package {kind} SHA-256 binding mismatch")
        if record.get("passed") is not True:
            errors.append(f"package {kind} record passed is not true")
        content = record.get("content_check")
        if not isinstance(content, Mapping) or content.get("passed") is not True:
            errors.append(f"package {kind} content check is not true")
        check_wrapper = record.get("check")
        if not isinstance(check_wrapper, Mapping) or check_wrapper.get("passed") is not True:
            errors.append(f"package {kind} install check is not true")
            continue
        check = check_wrapper
        nested_check = check_wrapper.get("check")
        if isinstance(nested_check, Mapping) and "module_paths" in nested_check:
            check = nested_check
        if check.get("passed") is not True:
            errors.append(f"package {kind} installed child check is not true")
            continue
        if check.get("source_checkout_in_import_path") is not False:
            errors.append(f"package {kind} imports from source checkout")
        install_root_value = check.get("install_root") or record.get("install_root")
        module_paths = check.get("module_paths")
        if not isinstance(install_root_value, str) or not install_root_value:
            errors.append(f"package {kind} install root is missing")
        if not isinstance(module_paths, Mapping) or not module_paths:
            errors.append(f"package {kind} module paths are missing")
        elif isinstance(install_root_value, str):
            install_root = Path(install_root_value).resolve()
            for module_name, module_path in module_paths.items():
                if not isinstance(module_name, str) or not isinstance(module_path, str):
                    errors.append(f"package {kind} module path is malformed")
                    continue
                try:
                    Path(module_path).resolve().relative_to(install_root)
                except ValueError:
                    errors.append(f"package {kind} module path escapes install root: {module_name}")
    return sorted(set(errors))


def _read_package_member(archive: Path, member_name: str) -> bytes:
    from robosuite.scripts import shakebench_verify_evidence_archive as archive_verifier

    names, _, _ = archive_verifier._archive_members(archive)
    if member_name not in names:
        raise ValueError(f"evidence archive is missing package evidence member: {member_name}")
    if archive_verifier._archive_kind(archive) == "zstd":
        completed = subprocess.run(
            ["tar", "--zstd", "-xOf", str(archive), member_name],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError("cannot safely read package evidence from zstd archive")
        return completed.stdout
    try:
        with tarfile.open(archive, "r:*") as stream:
            for member in stream.getmembers():
                if archive_verifier.normalize_member(member.name) == member_name:
                    if not member.isfile():
                        raise ValueError("package evidence archive member is not a regular file")
                    extracted = stream.extractfile(member)
                    if extracted is None:
                        break
                    return extracted.read()
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"cannot read package evidence archive: {exc}") from exc
    raise ValueError(f"cannot read package evidence member: {member_name}")


def _load_package_evidence(
    explicit_path: Path | None,
    archive_path: Path | None,
    member_name: str,
) -> tuple[dict[str, Any] | None, bytes | None, list[str]]:
    errors: list[str] = []
    sources: list[bytes] = []
    if explicit_path is not None:
        if not explicit_path.is_file():
            errors.append("package evidence file is missing")
        else:
            try:
                sources.append(explicit_path.read_bytes())
            except OSError as exc:
                errors.append(f"package evidence file is unreadable: {exc}")
    if archive_path is not None:
        if not archive_path.is_file():
            errors.append("package evidence archive is missing")
        else:
            try:
                sources.append(_read_package_member(archive_path, member_name))
            except (OSError, ValueError, tarfile.TarError) as exc:
                errors.append(str(exc))
    if not sources:
        return None, None, errors
    if len(sources) == 2 and sources[0] != sources[1]:
        errors.append("explicit and archive package evidence bytes differ")
    raw = sources[0]
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"package evidence is not valid JSON: {exc}")
        return None, raw, errors
    if not isinstance(value, dict):
        errors.append("package evidence is not an object")
        return None, raw, errors
    return value, raw, errors


def _compare_package_metadata(expected: Mapping[str, Any], actual: Mapping[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    for kind in ("wheel", "sdist"):
        expected_filename, expected_bytes, expected_sha = _package_fields(expected, kind)
        row = actual.get(kind)
        if not isinstance(row, Mapping):
            errors.append(f"{label} package {kind} metadata is missing")
            continue
        if (
            row.get("filename") != expected_filename
            or row.get("bytes") != expected_bytes
            or row.get("sha256") != expected_sha
        ):
            errors.append(f"{label} package {kind} metadata mismatch")
    if actual.get("sha256") != expected.get("sha256"):
        errors.append(f"{label} package evidence hash mismatch")
    return errors


def _verify_index_and_release(
    repo: Path,
    r6_manifest: Mapping[str, Any],
    release_path: Path,
    r6_authority: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        index = _read_json(repo / INDEX_FILENAME, "repository content index")
    except ValueError as exc:
        return [str(exc)]
    if index.get("schema_id") != "shakebench.evidence.index" or index.get("schema_version") != 1:
        errors.append("repository content index schema mismatch")
    copied = dict(index)
    expected = copied.pop("index_sha256", None)
    if expected != _sha256_bytes(_canonical(copied)):
        errors.append("repository content index self-hash mismatch")
    if index.get("archive_sha256") not in (None, ""):
        errors.append("R6 repository content index must not contain an outer archive hash")
    release = r6_manifest.get("release")
    if not isinstance(release, Mapping):
        errors.append("R6 manifest release binding is missing")
        release = {}
    expected_release = {
        "tag": R6_TAG,
        "archive_filename": R6_ARCHIVE_FILENAME,
        "archive_url": R6_ARCHIVE_URL,
        "content_index": INDEX_FILENAME,
        "history_rewrite_map": MAP_FILENAME,
        "removed_path_list": REMOVED_PATH_FILENAME,
    }
    for key, value in expected_release.items():
        if release.get(key) != value:
            errors.append(f"R6 manifest release field mismatch: {key}")
    if _sha256(repo / INDEX_FILENAME) != R6_INDEX_SHA256:
        errors.append("R6 repository content index hash is not frozen")
    if _sha256(repo / MAP_FILENAME) != R6_MAP_SHA256:
        errors.append("R6 history rewrite map hash is not frozen")
    if _sha256(repo / REMOVED_PATH_FILENAME) != R6_REMOVED_PATHS_SHA256:
        errors.append("R6 removed-path list hash is not frozen")
    if index.get("history_rewrite_map_sha256") != _sha256(repo / MAP_FILENAME):
        errors.append("content index/rewrite map hash mismatch")
    if index.get("path_list_sha256") != _sha256(repo / REMOVED_PATH_FILENAME):
        errors.append("content index/removed path list hash mismatch")
    map_asset = repo / RUNTIME_MAP_FILENAME
    if not map_asset.is_file() or map_asset.read_bytes() != (repo / MAP_FILENAME).read_bytes():
        errors.append("runtime/package rewrite map differs from repository map")
    runtime_contract = repo / "robosuite/models/assets/shakebench_runtime_contract.json"
    if not runtime_contract.is_file():
        errors.append("runtime publication contract is missing")
    else:
        try:
            contract = _read_json(runtime_contract, "runtime publication contract")
            if contract.get("history_rewrite", {}).get("map_sha256") != _sha256(repo / MAP_FILENAME):
                errors.append("runtime/repository rewrite map hash mismatch")
            if contract.get("compact_assets", {}).get(Path(RUNTIME_MAP_FILENAME).name) != _sha256(map_asset):
                errors.append("runtime compact rewrite map binding mismatch")
        except ValueError as exc:
            errors.append(str(exc))
    if not release_path.is_file():
        errors.append("detached R6 release manifest is missing")
        return errors
    try:
        detached = _read_json(release_path, "detached R6 release manifest")
    except ValueError as exc:
        return errors + [str(exc)]
    if detached.get("schema_id") != "shakebench.evidence.release_manifest" or detached.get("schema_version") != 1:
        errors.append("detached R6 release manifest schema mismatch")
    actual_r6_manifest_sha = _sha256(repo / R6_MANIFEST_FILENAME)
    if detached.get("r6_manifest_sha256") != actual_r6_manifest_sha:
        errors.append("detached R6 manifest SHA-256 binding mismatch")
    if actual_r6_manifest_sha != r6_authority.get("r6_manifest_sha256") or actual_r6_manifest_sha != R6_MANIFEST_SHA256:
        errors.append("R6 manifest SHA-256 is not frozen")
    if _sha256(release_path) != r6_authority.get("detached_release_manifest_sha256"):
        errors.append("detached R6 release manifest bytes are not frozen")
    archive = detached.get("archive")
    if not isinstance(archive, Mapping):
        errors.append("detached R6 archive binding is missing")
        archive = {}
    for key, value in (("filename", R6_ARCHIVE_FILENAME), ("sha256", R6_ARCHIVE_SHA256)):
        if archive.get(key) != value:
            errors.append(f"detached R6 archive field mismatch: {key}")
    if archive.get("byte_size") != 80628665:
        errors.append("detached R6 archive byte-size mismatch")
    if detached.get("release_tag") != R6_TAG or detached.get("release_url") != R6_ARCHIVE_URL:
        errors.append("detached R6 release tag/URL mismatch")
    if detached.get("embedded_content_index_sha256") != _sha256(repo / INDEX_FILENAME):
        errors.append("detached release/content index hash mismatch")
    if detached.get("history_rewrite_map_sha256") != _sha256(repo / MAP_FILENAME):
        errors.append("detached release/rewrite map hash mismatch")
    if detached.get("removed_path_list_sha256") != _sha256(repo / REMOVED_PATH_FILENAME):
        errors.append("detached release/removed path hash mismatch")
    r6_final = r6_manifest.get("final_commit")
    if r6_final != R6_IMPLEMENTATION_COMMIT or detached.get("r6_implementation_commit") != r6_final:
        errors.append("detached release/R6 implementation commit mismatch")
    expected_handoff = detached.get("expected_handoff")
    if not isinstance(expected_handoff, Mapping) or expected_handoff.get("implementation_commit") != r6_final:
        errors.append("detached release expected handoff implementation mismatch")
    if not isinstance(expected_handoff, Mapping) or expected_handoff.get("rule") != r6_manifest.get("handoff_rule"):
        errors.append("detached release expected handoff rule mismatch")
    package = r6_manifest.get("package_evidence")
    detached_package = detached.get("package_evidence")
    if not isinstance(package, Mapping) or not isinstance(detached_package, Mapping):
        errors.append("R6 package evidence binding is missing")
    else:
        errors.extend(_compare_package_metadata(package, detached_package, "detached R6 release"))
        expected_path = f"provenance/{Path(str(package.get('path', ''))).name}"
        if detached_package.get("path") != expected_path:
            errors.append("detached R6 package evidence path mismatch")
    return sorted(set(errors))


def _parse_sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not HEX64.fullmatch(fields[0]):
            raise ValueError(f"malformed SHA256SUMS line: {line}")
        values[fields[1].lstrip(" *") if fields[1].startswith((" ", "*")) else fields[1]] = fields[0]
    return values


def _verify_gate_release(
    repo: Path, manifest_file: Path, manifest: Mapping[str, Any], r6_authority: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    gate = manifest.get("gate_release")
    if not isinstance(gate, Mapping):
        return ["R6.1 gate release binding is missing"]
    gate_path_value = gate.get("manifest_path", GATE_RELEASE_MANIFEST_FILENAME)
    sums_path_value = gate.get("sha256sums_path", GATE_SUMS_FILENAME)
    try:
        gate_path = _safe_repo_path(repo, str(gate_path_value), "gate release manifest path")
        sums_path = _safe_repo_path(repo, str(sums_path_value), "gate SHA256SUMS path")
    except ValueError as exc:
        return [str(exc)]
    if not gate_path.is_file() or not sums_path.is_file():
        return ["R6.1 gate release metadata/checksums are missing"]
    try:
        gate_manifest = _read_json(gate_path, "R6.1 gate release manifest")
        sums = _parse_sums(sums_path)
    except (ValueError, OSError) as exc:
        return [str(exc)]
    if gate_manifest.get("schema_id") != "shakebench.gate.release_manifest" or gate_manifest.get("schema_version") != 1:
        errors.append("R6.1 gate release manifest schema mismatch")
    if gate_manifest.get("gate_tag") != gate.get("tag", R6_1_TAG):
        errors.append("R6.1 gate release tag binding mismatch")
    if _sha256(manifest_file) != gate_manifest.get("r6_1_manifest_sha256"):
        errors.append("R6.1 gate release/manifest hash mismatch")
    if gate_manifest.get("r6_authority") != {
        "release_tag": r6_authority.get("release_tag"),
        "archive_sha256": r6_authority.get("archive_sha256"),
        "archive_url": r6_authority.get("archive_url"),
    }:
        errors.append("R6.1 gate release/R6 authority mismatch")
    package = manifest.get("package_evidence")
    gate_package = gate_manifest.get("package_evidence")
    if not isinstance(package, Mapping) or not isinstance(gate_package, Mapping):
        errors.append("R6.1 gate package evidence binding is missing")
    else:
        errors.extend(_compare_package_metadata(package, gate_package, "R6.1 gate release"))
        if gate_package.get("asset_filename") != "package_evidence_final.json":
            errors.append("R6.1 gate package evidence asset filename mismatch")
    required_sum_names = {
        "phase_07_r6_1_manifest.json": _sha256(manifest_file),
        gate_path.name: _sha256(gate_path),
    }
    if isinstance(package, Mapping) and isinstance(package.get("sha256"), str):
        required_sum_names["package_evidence_final.json"] = package["sha256"]
    for name, expected in required_sum_names.items():
        if sums.get(name) != expected:
            errors.append(f"R6.1 SHA256SUMS binding mismatch: {name}")
    return sorted(set(errors))


def _verify_manifest_basics(repo: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("phase") != "07R6.1":
        errors.append("manifest phase mismatch")
    status = manifest.get("status")
    authorized = manifest.get("phase08_authorized")
    final_commit = manifest.get("final_commit")
    if status == "PASS":
        if authorized is not True:
            errors.append("PASS manifest is not phase08_authorized")
        if not isinstance(final_commit, str) or not HEX40.fullmatch(final_commit):
            errors.append("PASS manifest final_commit is missing or malformed")
        elif not _git_exists(repo, final_commit):
            errors.append("PASS manifest final_commit does not exist")
    elif status == "BLOCKED_BY_PHASE_07R6_1_GATE_HARDENING":
        if authorized is not False or final_commit is not None:
            errors.append("blocked manifest must keep phase08_authorized=false and final_commit=null")
        errors.append("Phase 07R6.1 manifest is intentionally blocked; authorization is not granted")
    else:
        errors.append("manifest status is not a Phase 07R6.1 status")
    science = manifest.get("protected_science")
    expected_science = {
        "controller_profile_id": "shakebench.reference_oracle.v3",
        "controller_profile_sha256": "60a5d351913a48d64480ea004eaabb2f91ea7f40add4968fefbeafb3a958991c",
        "official_physics_profile_id": "shakebench.official.physics.v2",
        "official_physics_profile_sha256": "c32d3962e62a9b9fc27b0de6bf787d8bf49ee17e306d6fbea9e480062a99606c",
        "dev_state_asset_sha256": "07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6",
    }
    if science != expected_science:
        errors.append("protected science hashes changed")
    anchor = manifest.get("dev_state_anchor")
    if not isinstance(anchor, Mapping):
        errors.append("manifest dev-state anchor is missing")
    elif (
        anchor.get("pre_history_rewrite_commit") != "dd6fe2edb6384ccdb5116be44f07592b4864e377"
        or anchor.get("rewritten_commit") != "08626ea5a5e107df503e266be9065b929d47f882"
        or anchor.get("asset_path") != DEV_ASSET
        or anchor.get("asset_sha256") != expected_science["dev_state_asset_sha256"]
    ):
        errors.append("manifest dev-state anchor binding changed")
    authority = manifest.get("r6_authority")
    expected_authority = {
        "implementation_commit": R6_IMPLEMENTATION_COMMIT,
        "handoff_commit": R6_HANDOFF_COMMIT,
        "release_tag": R6_TAG,
        "r6_manifest_sha256": R6_MANIFEST_SHA256,
        "detached_release_manifest_sha256": R6_RELEASE_MANIFEST_SHA256,
        "archive_sha256": R6_ARCHIVE_SHA256,
        "archive_filename": R6_ARCHIVE_FILENAME,
        "archive_url": R6_ARCHIVE_URL,
        "content_index_sha256": R6_INDEX_SHA256,
        "history_rewrite_map_sha256": R6_MAP_SHA256,
        "removed_path_list_sha256": R6_REMOVED_PATHS_SHA256,
    }
    if authority != expected_authority:
        errors.append("R6 authority binding changed")
    return errors


def verify_handoff(
    manifest_path: str | Path = MANIFEST_FILENAME,
    rewrite_map_path: str | Path = MAP_FILENAME,
    *,
    repo_root: str | Path | None = None,
    release_manifest_path: str | Path = RELEASE_MANIFEST_FILENAME,
    package_evidence_path: str | Path | None = None,
    evidence_archive_path: str | Path | None = None,
    remote: str | None = None,
) -> dict[str, Any]:
    """Verify a Phase 07R6.1 handoff; PASS mode requires package authority."""

    manifest_file = Path(manifest_path).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else manifest_file.parents[1]
    map_file = Path(rewrite_map_path).resolve()
    release_file = Path(release_manifest_path).resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(manifest_file, "Phase 07R6.1 manifest")
        r6_manifest = _read_json(repo / R6_MANIFEST_FILENAME, "R6 manifest")
    except ValueError as exc:
        return {"passed": False, "errors": [str(exc)]}
    errors.extend(_verify_manifest_basics(repo, manifest))
    mapping, map_errors = _verify_map(repo, map_file, dict(r6_manifest))
    errors.extend(map_errors)
    r6_authority = manifest.get("r6_authority") if isinstance(manifest.get("r6_authority"), Mapping) else {}
    try:
        errors.extend(_verify_index_and_release(repo, r6_manifest, release_file, r6_authority))
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"R6 authority verification failed closed: {exc}")
    current_head = _git(repo, "rev-parse", "HEAD")
    errors.extend(
        _verify_tag(
            repo,
            R6_TAG,
            str(r6_authority.get("handoff_commit")),
            current_head,
            (R6_MANIFEST_FILENAME, RELEASE_MANIFEST_FILENAME, INDEX_FILENAME, MAP_FILENAME, REMOVED_PATH_FILENAME),
            remote=remote,
        )
    )
    status = manifest.get("status")
    handoff_commit: str | None = None
    if (
        status == "PASS"
        and isinstance(manifest.get("final_commit"), str)
        and _git_exists(repo, manifest["final_commit"])
    ):
        final_commit = manifest["final_commit"]
        pre_head = manifest.get("pre_implementation_head")
        if not isinstance(pre_head, str) or not HEX40.fullmatch(pre_head):
            errors.append("pre-implementation HEAD is missing or malformed")
        else:
            errors.extend(
                _verify_commit_range(
                    repo,
                    R6_HANDOFF_COMMIT,
                    pre_head,
                    {"docs/phase_07_report.md"},
                    "R6 handoff to R6.1 base",
                )
            )
            errors.extend(
                _verify_commit_range(
                    repo,
                    pre_head,
                    final_commit,
                    manifest.get("allowed_implementation_files", ()),
                    "R6.1 implementation",
                )
            )
        handoff_commit = _git(repo, "log", "-1", "--format=%H", "--", str(manifest_file.relative_to(repo)))
        if not HEX40.fullmatch(handoff_commit):
            errors.append("R6.1 manifest has no Git handoff commit")
        else:
            errors.extend(
                _verify_commit_range(
                    repo,
                    final_commit,
                    handoff_commit,
                    manifest.get("allowed_handoff_files", ()),
                    "R6.1 metadata handoff",
                )
            )
            if handoff_commit == final_commit:
                errors.append("R6.1 manifest is not in a distinct metadata handoff commit")
            if current_head != handoff_commit:
                errors.extend(
                    _verify_commit_range(
                        repo,
                        handoff_commit,
                        current_head,
                        manifest.get("post_handoff_allowed_files", DEFAULT_R6_1_POST_HANDOFF_FILES),
                        "post-handoff",
                    )
                )
                if not manifest.get("post_handoff_allowed_files"):
                    errors.append("current HEAD is not the R6.1 metadata handoff commit")
        errors.extend(_verify_gate_release(repo, manifest_file, manifest, r6_authority))
        gate = manifest.get("gate_release")
        gate_tag = gate.get("tag") if isinstance(gate, Mapping) else None
        if not isinstance(gate_tag, str):
            errors.append("R6.1 gate tag is missing")
        else:
            expected_gate_commit = handoff_commit or current_head
            errors.extend(
                _verify_tag(
                    repo,
                    gate_tag,
                    expected_gate_commit,
                    current_head,
                    (MANIFEST_FILENAME, GATE_RELEASE_MANIFEST_FILENAME, GATE_SUMS_FILENAME),
                    remote=remote,
                )
            )
        package = manifest.get("package_evidence")
        explicit = Path(package_evidence_path).resolve() if package_evidence_path is not None else None
        archive = Path(evidence_archive_path).resolve() if evidence_archive_path is not None else None
        if explicit is None and archive is None:
            errors.append("PASS/authorization mode requires explicit package evidence or evidence archive")
        member_name = "provenance/package_evidence_final.json"
        package_payload, package_bytes, package_errors = _load_package_evidence(explicit, archive, member_name)
        errors.extend(package_errors)
        if package_payload is not None and isinstance(package, Mapping):
            errors.extend(validate_package_evidence(package_payload, package))
            if package_bytes is not None and _sha256_bytes(package_bytes) != package.get("sha256"):
                errors.append("package evidence file hash mismatch")
        elif package_payload is not None:
            errors.append("manifest package evidence binding is missing")
    elif status == "PASS":
        errors.append("PASS handoff cannot be checked because final_commit is invalid")
    result = {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "manifest": str(manifest_file),
        "head": current_head,
        "final_commit": manifest.get("final_commit"),
        "handoff_commit": handoff_commit,
        "r6_tag_commit": _tag_commit(repo, R6_TAG),
        "rewritten_anchor": mapping.get("rewritten_anchor") if mapping else None,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(MANIFEST_FILENAME))
    parser.add_argument("--rewrite-map", type=Path, default=Path(MAP_FILENAME))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--release-manifest", type=Path, default=Path(RELEASE_MANIFEST_FILENAME))
    package = parser.add_mutually_exclusive_group()
    package.add_argument("--package-evidence", type=Path, default=None)
    package.add_argument("--evidence-archive", type=Path, default=None)
    parser.add_argument("--remote", default=None, help="remote name to verify peeled tag refs, e.g. origin")
    args = parser.parse_args(argv)
    result = verify_handoff(
        args.manifest,
        args.rewrite_map,
        repo_root=args.repo_root,
        release_manifest_path=args.release_manifest,
        package_evidence_path=args.package_evidence,
        evidence_archive_path=args.evidence_archive,
        remote=args.remote,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_package_evidence", "verify_handoff", "main"]
