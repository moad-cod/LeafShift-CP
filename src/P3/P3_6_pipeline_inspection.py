
#!/usr/bin/env python3
"""
P3.6 — Inspect exact functions needed for adaptive CQR.

Read-only source-code inspection.
No models are fitted, no predictions are made,
and no research files are modified.
"""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "src/E3/run_cqr_baselines.py",
    ROOT / "src/P1/p11_u5_prediction_export.py",
    ROOT / "src/P0/cqr_clipping_sensitivity.py",
]

IMPORTANT_TERMS = [
    "load",
    "prepare",
    "feature",
    "weight",
    "quantile",
    "split",
    "fold",
    "run",
    "fit",
    "main",
]

print("=" * 80)
print("P3.6 — ORIGINAL CQR PIPELINE INSPECTION")
print("=" * 80)

for path in FILES:
    print("\n" + "=" * 80)
    print(f"FILE: {path.relative_to(ROOT)}")
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    source = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print("SYNTAX ERROR:", exc)
        continue

    functions = [
        node for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
    ]

    print("\nALL TOP-LEVEL FUNCTIONS:")

    for node in functions:
        print(
            f"  {node.name} "
            f"(lines {node.lineno}-{node.end_lineno})"
        )

    print("\nRELEVANT FUNCTION DEFINITIONS:")

    for node in functions:
        name = node.name.lower()

        if not any(term in name for term in IMPORTANT_TERMS):
            continue

        print("\n" + "-" * 70)
        print(
            f"FUNCTION: {node.name} "
            f"(lines {node.lineno}-{node.end_lineno})"
        )
        print("-" * 70)

        # Print function signature plus limited body.
        start = node.lineno - 1
        end = min(node.end_lineno, start + 42)

        for i in range(start, end):
            print(f"{i+1:4}: {lines[i]}")

        if node.end_lineno > end:
            print(
                f"... {node.end_lineno-end} lines omitted"
            )

    print("\nTOP-LEVEL CONFIGURATION (first 55 lines):")

    for index, line in enumerate(lines[:55], 1):
        print(f"{index:4}: {line}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("No files were modified.")
print("=" * 80)
