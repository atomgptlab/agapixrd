#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


def scalar_preview(x, max_len=120):
    s = repr(x)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def type_name(x):
    if isinstance(x, dict):
        return "dict"
    if isinstance(x, list):
        return "list"
    if isinstance(x, str):
        return "str"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if x is None:
        return "None"
    return type(x).__name__


def summarize_list(xs, path, depth, max_depth, max_items):
    indent = "  " * depth
    print(f"{indent}{path}: list len={len(xs)}")

    if not xs:
        return

    type_counts = Counter(type_name(x) for x in xs)
    print(f"{indent}  element types: {dict(type_counts)}")

    for i, item in enumerate(xs[:max_items]):
        walk(item, f"{path}[{i}]", depth + 1, max_depth, max_items)

    if len(xs) > max_items:
        print(f"{indent}  ... {len(xs) - max_items} more items")


def summarize_dict(d, path, depth, max_depth, max_items):
    indent = "  " * depth
    keys = list(d.keys())

    print(f"{indent}{path}: dict len={len(keys)}")

    if not keys:
        return

    print(f"{indent}  keys preview: {keys[:max_items]}")
    if len(keys) > max_items:
        print(f"{indent}  ... {len(keys) - max_items} more keys")

    for k in keys[:max_items]:
        v = d[k]
        child_path = f"{path}.{k}" if path else str(k)
        walk(v, child_path, depth + 1, max_depth, max_items)


def walk(x, path="$", depth=0, max_depth=4, max_items=5):
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}{path}: {type_name(x)} ... max_depth reached")
        return

    if isinstance(x, dict):
        summarize_dict(x, path, depth, max_depth, max_items)
    elif isinstance(x, list):
        summarize_list(x, path, depth, max_depth, max_items)
    else:
        print(f"{indent}{path}: {type_name(x)} = {scalar_preview(x)}")


def find_poscar_like_strings(x, path="$", hits=None, max_hits=30):
    """
    Search for long strings that look like POSCAR/CIF/structure payloads.
    """
    if hits is None:
        hits = []

    if len(hits) >= max_hits:
        return hits

    if isinstance(x, dict):
        for k, v in x.items():
            find_poscar_like_strings(v, f"{path}.{k}", hits, max_hits)
            if len(hits) >= max_hits:
                break

    elif isinstance(x, list):
        for i, v in enumerate(x):
            find_poscar_like_strings(v, f"{path}[{i}]", hits, max_hits)
            if len(hits) >= max_hits:
                break

    elif isinstance(x, str):
        s = x.strip()
        lines = s.splitlines()

        looks_like_poscar = (
            len(lines) >= 8
            and any(tok in s for tok in ["Direct", "Cartesian", "Selective dynamics"])
        )

        looks_like_cif = (
            "_cell_length_a" in s
            or "_atom_site" in s
            or "data_" in lines[0].lower() if lines else False
        )

        if looks_like_poscar or looks_like_cif:
            hits.append(
                {
                    "path": path,
                    "kind": "POSCAR-like" if looks_like_poscar else "CIF-like",
                    "n_lines": len(lines),
                    "preview": "\n".join(lines[:12]),
                }
            )

    return hits


def find_key_paths(x, query_terms, path="$", hits=None, max_hits=100):
    """
    Search paths whose key names contain useful words like poscar, structure, pred, gt.
    """
    if hits is None:
        hits = []

    if len(hits) >= max_hits:
        return hits

    if isinstance(x, dict):
        for k, v in x.items():
            child_path = f"{path}.{k}"
            key_lower = str(k).lower()

            if any(term in key_lower for term in query_terms):
                hits.append(
                    {
                        "path": child_path,
                        "key": k,
                        "value_type": type_name(v),
                        "value_preview": scalar_preview(v) if not isinstance(v, (dict, list)) else "",
                    }
                )

            find_key_paths(v, query_terms, child_path, hits, max_hits)

            if len(hits) >= max_hits:
                break

    elif isinstance(x, list):
        for i, v in enumerate(x):
            find_key_paths(v, query_terms, f"{path}[{i}]", hits, max_hits)
            if len(hits) >= max_hits:
                break

    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--max-hits", type=int, default=40)
    args = parser.parse_args()

    print(f"Reading: {args.json_path}")

    with args.json_path.open() as f:
        data = json.load(f)

    print("\n" + "=" * 100)
    print("TREE SHAPE")
    print("=" * 100)
    walk(data, max_depth=args.max_depth, max_items=args.max_items)

    print("\n" + "=" * 100)
    print("KEY PATHS THAT LOOK STRUCTURE-RELATED")
    print("=" * 100)

    key_hits = find_key_paths(
        data,
        query_terms=[
            "poscar",
            "structure",
            "cif",
            "pred",
            "gt",
            "ground",
            "truth",
            "target",
            "refined",
            "initial",
            "final",
            "id",
            "jid",
            "entry",
        ],
        max_hits=args.max_hits,
    )

    if not key_hits:
        print("No structure-ish key paths found.")
    else:
        for h in key_hits:
            print(f"{h['path']}    key={h['key']}    type={h['value_type']}")
            if h["value_preview"]:
                print(f"  preview: {h['value_preview']}")

    print("\n" + "=" * 100)
    print("POSCAR/CIF-LIKE STRING PAYLOADS")
    print("=" * 100)

    poscar_hits = find_poscar_like_strings(data, max_hits=args.max_hits)

    if not poscar_hits:
        print("No obvious POSCAR/CIF-like strings found.")
    else:
        for h in poscar_hits:
            print(f"\nPATH: {h['path']}")
            print(f"KIND: {h['kind']}")
            print(f"LINES: {h['n_lines']}")
            print("PREVIEW:")
            print(h["preview"])


if __name__ == "__main__":
    main()
