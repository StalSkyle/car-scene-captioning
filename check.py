#!/usr/bin/env python3
"""Architecture/convention checker for car-scene-captioning.

Runs without third-party dependencies. It checks project structure statically with AST,
so ML weights do not need to be present and models are never imported/loaded.

Usage:
    python check.py
    python check.py --json
    python check.py --root /path/to/repository
    python check.py --strict   # return exit code 1 when violations are found
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
AUTOCAPTION = ROOT / "autocaption"

RULES = {
    "R1": "main.py is orchestration-only (no direct ML framework/model loading)",
    "R2": "main.py uses project code only through the public autocaption API",
    "R3": "symbols imported from autocaption are declared in autocaption.__all__",
    "R4": "autocaption is self-contained and does not import project-root helpers",
    "R5": "heavy pipeline components are initialized outside image-processing loops",
    "R6": "new image path/URL loading goes through ImageLoader (legacy direct loads are allowlisted)",
    "R7": "local model files are loaded from MODELS/",
}

HEAVY_ML_MODULES = {"torch", "torchvision", "transformers", "ultralytics"}
IMAGE_LOADER_NAME = "ImageLoader"
LOCAL_MODEL_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    message: str


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_python(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None


def production_files() -> list[Path]:
    files: list[Path] = []
    main = ROOT / "main.py"
    if main.exists():
        files.append(main)
    if AUTOCAPTION.exists():
        files.extend(sorted(AUTOCAPTION.rglob("*.py")))
    return files


def root_python_modules() -> set[str]:
    """Top-level project-local .py modules that are not entry/check scripts."""
    ignored = {"main", "check", "setup", "conftest"}
    return {
        p.stem
        for p in ROOT.glob("*.py")
        if p.stem not in ignored and not p.stem.startswith("test_")
    }


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def import_top_level(node: ast.AST) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            out.append((alias.name.split(".")[0], node.lineno))
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        out.append((node.module.split(".")[0], node.lineno))
    return out


def check_r1_main_orchestration() -> list[Violation]:
    path = ROOT / "main.py"
    tree = parse_python(path)
    if tree is None:
        return []

    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for top, line in import_top_level(node):
                if top in HEAVY_ML_MODULES:
                    found.append(Violation(
                        "R1", rel(path), line,
                        f"main.py imports heavy ML module '{top}' directly",
                    ))
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func) or ""
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "YOLO" or name == "torch.load" or leaf == "from_pretrained":
                found.append(Violation(
                    "R1", rel(path), getattr(node, "lineno", 0),
                    f"main.py initializes/loads a model directly via '{name}'",
                ))
    return found


def check_r2_main_public_api() -> list[Violation]:
    path = ROOT / "main.py"
    tree = parse_python(path)
    if tree is None:
        return []

    root_modules = root_python_modules()
    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("autocaption."):
                found.append(Violation(
                    "R2", rel(path), node.lineno,
                    f"bypasses public API with 'from {node.module} import ...'",
                ))
            elif node.module.split(".")[0] in root_modules:
                found.append(Violation(
                    "R2", rel(path), node.lineno,
                    f"imports project-root helper '{node.module}' instead of public autocaption API",
                ))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("autocaption."):
                    found.append(Violation(
                        "R2", rel(path), node.lineno,
                        f"bypasses public API with 'import {alias.name}'",
                    ))
                elif alias.name.split(".")[0] in root_modules:
                    found.append(Violation(
                        "R2", rel(path), node.lineno,
                        f"imports project-root helper '{alias.name}' instead of public autocaption API",
                    ))
    return found


def extract_all_names(path: Path) -> set[str]:
    tree = parse_python(path)
    if tree is None:
        return set()
    result: set[str] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            result.add(elt.value)
    return result


def check_r3_exports() -> list[Violation]:
    main = ROOT / "main.py"
    init = AUTOCAPTION / "__init__.py"
    tree = parse_python(main)
    if tree is None or not init.exists():
        return []
    exported = extract_all_names(init)
    if not exported:
        # If the project stops using __all__, do not invent violations.
        return []

    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "autocaption":
            for alias in node.names:
                if alias.name != "*" and alias.name not in exported:
                    found.append(Violation(
                        "R3", rel(main), node.lineno,
                        f"'{alias.name}' is imported from autocaption but missing from autocaption.__all__",
                    ))
    return found


def check_r4_package_self_contained() -> list[Violation]:
    if not AUTOCAPTION.exists():
        return []
    root_modules = root_python_modules()
    found: list[Violation] = []

    for path in sorted(AUTOCAPTION.rglob("*.py")):
        tree = parse_python(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and node.module.split(".")[0] in root_modules:
                    found.append(Violation(
                        "R4", rel(path), node.lineno,
                        f"package module imports project-root helper '{node.module}'; keep shared pipeline code inside autocaption",
                    ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in root_modules:
                        found.append(Violation(
                            "R4", rel(path), node.lineno,
                            f"package module imports project-root helper '{alias.name}'; keep shared pipeline code inside autocaption",
                        ))
    return found


def autocaption_imported_names(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "autocaption":
            result.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return result


def calls_inside(node: ast.AST, names: set[str]) -> Iterable[ast.Call]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in names:
            yield child


def check_r5_heavy_init_outside_loops() -> list[Violation]:
    path = ROOT / "main.py"
    tree = parse_python(path)
    if tree is None:
        return []

    component_names = autocaption_imported_names(tree) - {IMAGE_LOADER_NAME}
    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for call in calls_inside(node, component_names):
                found.append(Violation(
                    "R5", rel(path), call.lineno,
                    f"pipeline component '{call.func.id}' is initialized inside a processing loop",
                ))
    return found


# The original project already contains two direct Image.open() calls outside
# ImageLoader. They are technical debt, not violations introduced by the agent.
# R6 therefore enforces an incremental rule: do not add NEW direct image-loading
# calls outside autocaption/image_loader.py.
LEGACY_IMAGE_LOADING_ALLOWANCE = {
    "autocaption/feature_extractor.py": {"Image.open": 1},
    "autocaption/text_generator.py": {"Image.open": 1},
}


def check_r6_image_loading_boundary() -> list[Violation]:
    found: list[Violation] = []
    interesting = {
        "requests.get",
        "requests.post",
        "requests.request",
        "Image.open",
        "PIL.Image.open",
    }

    for path in production_files():
        if path.resolve() == (AUTOCAPTION / "image_loader.py").resolve():
            continue

        tree = parse_python(path)
        if tree is None:
            continue

        calls_by_name: dict[str, list[int]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func) or ""
            if name in interesting:
                calls_by_name.setdefault(name, []).append(node.lineno)

        relative = rel(path)
        allowance = LEGACY_IMAGE_LOADING_ALLOWANCE.get(relative, {})

        for name, lines in calls_by_name.items():
            allowed = allowance.get(name, 0)
            extra = max(0, len(lines) - allowed)
            if extra == 0:
                continue

            for line in sorted(lines)[allowed:]:
                found.append(Violation(
                    "R6", relative, line,
                    f"new image loading/network fetch '{name}' is outside "
                    "autocaption/image_loader.py "
                    f"(legacy allowance for this file: {allowed})",
                ))

    return found


def literal_string_arg(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def is_models_path(value: str) -> bool:
    norm = value.replace("\\", "/").lstrip("./")
    return norm == "MODELS" or norm.startswith("MODELS/")


def clearly_local_pretrained_path(value: str) -> bool:
    p = value.replace("\\", "/")
    suffix = Path(p).suffix.lower()
    return (
        p.startswith(("./", "../", "/", "MODELS/", "models/"))
        or (len(p) >= 3 and p[1:3] == ":/")
        or suffix in LOCAL_MODEL_EXTENSIONS
    )


def check_r7_model_paths() -> list[Violation]:
    found: list[Violation] = []
    for path in production_files():
        tree = parse_python(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func) or ""
            leaf = name.rsplit(".", 1)[-1]
            value = literal_string_arg(node)
            if value is None:
                continue

            must_be_models = leaf == "YOLO" or name == "torch.load"
            if leaf == "from_pretrained" and clearly_local_pretrained_path(value):
                must_be_models = True

            if must_be_models and not is_models_path(value):
                found.append(Violation(
                    "R7", rel(path), node.lineno,
                    f"local model path '{value}' is outside MODELS/",
                ))
    return found


def run_checks() -> list[Violation]:
    checks = [
        check_r1_main_orchestration,
        check_r2_main_public_api,
        check_r3_exports,
        check_r4_package_self_contained,
        check_r5_heavy_init_outside_loops,
        check_r6_image_loading_boundary,
        check_r7_model_paths,
    ]
    violations: list[Violation] = []
    for check in checks:
        violations.extend(check())
    return sorted(violations, key=lambda v: (v.rule, v.path, v.line, v.message))


def print_human(violations: list[Violation]) -> None:
    by_rule = {rule: [] for rule in RULES}
    for violation in violations:
        by_rule[violation.rule].append(violation)

    print("Convention check")
    print("=" * 72)
    for rule, description in RULES.items():
        items = by_rule[rule]
        status = "PASS" if not items else f"FAIL ({len(items)})"
        print(f"[{rule}] {status} - {description}")
        for item in items:
            where = f"{item.path}:{item.line}" if item.line else item.path
            print(f"      {where}: {item.message}")
    print("=" * 72)
    print(f"Violations: {len(violations)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="project root (default: directory containing check.py)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if violations exist")
    args = parser.parse_args()

    global ROOT, AUTOCAPTION
    if args.root is not None:
        ROOT = args.root.resolve()
        AUTOCAPTION = ROOT / "autocaption"

    violations = run_checks()
    if args.json:
        payload = {
            "violations": len(violations),
            "items": [asdict(v) for v in violations],
            "rules": RULES,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(violations)

    return 1 if args.strict and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
