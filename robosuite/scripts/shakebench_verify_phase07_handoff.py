"""Verify the executable Phase 07R6 Git/publication handoff contract.

This verifier is intentionally repository-oriented.  It checks Git objects,
tree contents, ancestry, metadata-only handoff changes, and cross-file hashes;
it does not treat a 40/64-character string as proof that an object exists.
Outer archive bytes are checked by the standalone archive verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_FILENAME = "docs/phase_07_r6_manifest.json"
MAP_FILENAME = "docs/shakebench_history_rewrite_map_v2.json"
RUNTIME_MAP_FILENAME = "robosuite/models/assets/shakebench_history_rewrite_map_v2.json"
INDEX_FILENAME = "docs/shakebench_evidence_index_v1.json"
REMOVED_PATH_FILENAME = "docs/shakebench_evidence_removed_paths_v1.txt"
RELEASE_MANIFEST_FILENAME = "docs/shakebench_evidence_release_manifest_v1.json"
DEV_ASSET = "robosuite/models/assets/shakebench_states_dev.json"
ALLOWED_HANDOFF_FILES = {
    MANIFEST_FILENAME,
    "docs/phase_07_report.md",
    "docs/shakebench_prompts/README.md",
    INDEX_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    "SHA256SUMS",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
    return hashlib.sha256(completed.stdout).hexdigest()


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
    elif hashlib.sha256(_canonical(payload)).hexdigest() != expected_payload_hash:
        errors.append("history rewrite map payload hash mismatch")
    history = mapping.get("history_rewrite")
    if not isinstance(history, dict):
        errors.append("history rewrite details are missing")
    else:
        if history.get("target_refs") != ["refs/heads/master"]:
            errors.append("history rewrite target refs are not explicit")
        if history.get("path_list") != REMOVED_PATH_FILENAME:
            errors.append("history rewrite path-list binding mismatch")
        if history.get("path_list_sha256") != _sha256(repo / REMOVED_PATH_FILENAME):
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
        if role in seen_roles or role not in required_roles:
            errors.append("commit mapping role is missing or duplicated")
        seen_roles.add(str(role))
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
    elif _ancestor(repo, old_anchor, _git(repo, "rev-parse", "HEAD")):
        errors.append("pre-rewrite anchor is still reachable from current HEAD")
    if anchor.get("asset_path") != DEV_ASSET:
        errors.append("rewritten anchor asset path mismatch")
    if anchor.get("asset_sha256") != manifest.get("dev_state_anchor", {}).get("asset_sha256"):
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


def _verify_index_and_release(repo: Path, manifest: dict[str, Any], release_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        index = _read_json(repo / INDEX_FILENAME, "repository content index")
    except ValueError as exc:
        return [str(exc)]
    if index.get("schema_id") != "shakebench.evidence.index" or index.get("schema_version") != 1:
        errors.append("repository content index schema mismatch")
    copied = dict(index)
    expected = copied.pop("index_sha256", None)
    if expected != hashlib.sha256(_canonical(copied)).hexdigest():
        errors.append("repository content index self-hash mismatch")
    if index.get("archive_sha256") not in (None, ""):
        errors.append("R6 repository content index must not contain an outer archive hash")
    release = manifest.get("release")
    if not isinstance(release, dict):
        errors.append("manifest release binding is missing")
    else:
        if release.get("content_index") != INDEX_FILENAME:
            errors.append("manifest content-index path mismatch")
        if release.get("history_rewrite_map") != MAP_FILENAME:
            errors.append("manifest rewrite-map path mismatch")
        if release.get("removed_path_list") != REMOVED_PATH_FILENAME:
            errors.append("manifest removed-path path mismatch")
        if release.get("tag") != "shakebench-evidence-v0-2026-09-r6":
            errors.append("R6 release tag mismatch")
        if release.get("archive_filename") != "shakebench-evidence-v0-2026-09-r6.tar.zst":
            errors.append("R6 archive filename mismatch")
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
            if contract.get("compact_assets", {}).get(RUNTIME_MAP_FILENAME.rsplit("/", 1)[-1]) != _sha256(map_asset):
                errors.append("runtime compact rewrite map binding mismatch")
        except ValueError as exc:
            errors.append(str(exc))
    if not release_path.is_file():
        errors.append("detached release manifest is missing")
        return errors
    try:
        detached = _read_json(release_path, "detached release manifest")
    except ValueError as exc:
        return errors + [str(exc)]
    if detached.get("schema_id") != "shakebench.evidence.release_manifest" or detached.get("schema_version") != 1:
        errors.append("detached release manifest schema mismatch")
    if detached.get("release_tag") != (release or {}).get("tag"):
        errors.append("detached release/manifest tag mismatch")
    if detached.get("embedded_content_index_sha256") != _sha256(repo / INDEX_FILENAME):
        errors.append("detached release/content index hash mismatch")
    if detached.get("history_rewrite_map_sha256") != _sha256(repo / MAP_FILENAME):
        errors.append("detached release/rewrite map hash mismatch")
    if detached.get("removed_path_list_sha256") != _sha256(repo / REMOVED_PATH_FILENAME):
        errors.append("detached release/removed path hash mismatch")
    if detached.get("r6_implementation_commit") != manifest.get("final_commit"):
        errors.append("detached release/implementation commit mismatch")
    package = manifest.get("package_evidence")
    detached_package = detached.get("package_evidence")
    if not isinstance(package, dict) or not isinstance(detached_package, dict):
        errors.append("package evidence binding is missing")
    else:
        if package.get("sha256") != detached_package.get("sha256"):
            errors.append("package evidence hash mismatch")
        if package.get("wheel_sha256") != detached_package.get("wheel", {}).get("sha256"):
            errors.append("wheel hash binding mismatch")
        if package.get("sdist_sha256") != detached_package.get("sdist", {}).get("sha256"):
            errors.append("sdist hash binding mismatch")
    return errors


def verify_handoff(
    manifest_path: str | Path = MANIFEST_FILENAME,
    rewrite_map_path: str | Path = MAP_FILENAME,
    *,
    repo_root: str | Path | None = None,
    release_manifest_path: str | Path = RELEASE_MANIFEST_FILENAME,
    package_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root is not None else Path(manifest_path).resolve().parents[1]
    manifest_file = Path(manifest_path).resolve()
    map_file = Path(rewrite_map_path).resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(manifest_file, "Phase 07R6 manifest")
    except ValueError as exc:
        return {"passed": False, "errors": [str(exc)]}
    if manifest.get("phase") != "07R6":
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
    elif status == "BLOCKED_BY_PHASE_07R6_PUBLICATION_INTEGRITY":
        if authorized is not False or final_commit is not None:
            errors.append("blocked manifest must keep phase08_authorized=false and final_commit=null")
    else:
        errors.append("manifest status is not an R6 status")
    anchor = manifest.get("dev_state_anchor")
    if not isinstance(anchor, dict):
        errors.append("manifest dev-state anchor is missing")
    else:
        if anchor.get("pre_history_rewrite_commit") != "dd6fe2edb6384ccdb5116be44f07592b4864e377":
            errors.append("manifest pre-rewrite anchor mismatch")
        if anchor.get("rewritten_commit") != "08626ea5a5e107df503e266be9065b929d47f882":
            errors.append("manifest rewritten anchor mismatch")
        if (
            anchor.get("asset_path") != DEV_ASSET
            or anchor.get("asset_sha256") != "07de20b40b2500923f44c6b5edd179378ef7475a600752a08f1f7b000cd000c6"
        ):
            errors.append("manifest dev-state asset binding mismatch")
    mapping, map_errors = _verify_map(repo, map_file, manifest)
    errors.extend(map_errors)
    current_head = _git(repo, "rev-parse", "HEAD")
    if status == "PASS" and isinstance(final_commit, str) and _git_exists(repo, final_commit):
        if not _ancestor(repo, final_commit, current_head):
            errors.append("implementation commit is not an ancestor of handoff HEAD")
        handoff_commit = _git(repo, "log", "-1", "--format=%H", "--", str(manifest_file.relative_to(repo)))
        if not HEX40.fullmatch(handoff_commit):
            errors.append("manifest has no Git handoff commit")
        else:
            changed = set(_git(repo, "diff", "--name-only", final_commit, handoff_commit).splitlines())
            if not changed.issubset(ALLOWED_HANDOFF_FILES):
                errors.append(
                    "implementation-to-handoff diff contains non-metadata files: "
                    + ", ".join(sorted(changed - ALLOWED_HANDOFF_FILES))
                )
            if handoff_commit == final_commit:
                errors.append("manifest is not in a distinct metadata handoff commit")
    errors.extend(_verify_index_and_release(repo, manifest, Path(release_manifest_path).resolve()))
    package_path = Path(package_evidence_path).resolve() if package_evidence_path else None
    package = manifest.get("package_evidence")
    if package_path is not None:
        if not package_path.is_file():
            errors.append("package evidence file is missing")
        elif not isinstance(package, dict) or package.get("sha256") != _sha256(package_path):
            errors.append("package evidence file hash mismatch")
        else:
            try:
                package_payload = _read_json(package_path, "package evidence")
                if package_payload.get("passed") is not True:
                    errors.append("package evidence is not PASS")
                for kind, field in (("wheel", "wheel_sha256"), ("sdist", "sdist_sha256")):
                    records = [
                        row
                        for row in package_payload.get("records", [])
                        if isinstance(row, dict) and row.get("kind") == kind
                    ]
                    if len(records) != 1 or records[0].get("artifact_sha256") != package.get(field):
                        errors.append(f"package {kind} hash binding mismatch")
            except ValueError as exc:
                errors.append(str(exc))
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "manifest": str(manifest_file),
        "head": current_head,
        "final_commit": final_commit,
        "handoff_commit": (
            _git(repo, "log", "-1", "--format=%H", "--", str(manifest_file.relative_to(repo)))
            if manifest_file.is_relative_to(repo)
            else None
        ),
        "rewritten_anchor": mapping.get("rewritten_anchor") if mapping else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(MANIFEST_FILENAME))
    parser.add_argument("--rewrite-map", type=Path, default=Path(MAP_FILENAME))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--release-manifest", type=Path, default=Path(RELEASE_MANIFEST_FILENAME))
    parser.add_argument("--package-evidence", type=Path, default=None)
    args = parser.parse_args(argv)
    result = verify_handoff(
        args.manifest,
        args.rewrite_map,
        repo_root=args.repo_root,
        release_manifest_path=args.release_manifest,
        package_evidence_path=args.package_evidence,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["verify_handoff", "main"]
