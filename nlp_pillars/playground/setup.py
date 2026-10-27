"""Setup code: the imports a notebook uses but never wrote down.

A notebook saved after its import cell was deleted still uses ``np`` or
``IsolationForest`` everywhere, and every cell that does fails with
``NameError`` in a fresh kernel. :func:`suggest_setup` finds the names the
notebook's code cells load but never assign, define or import anywhere, and
turns the ones in :data:`KNOWN_IMPORTS` into import lines. A name that is not
in the table is never guessed at.

The check is notebook-wide and ignores scope: a name assigned anywhere in any
cell (a variable, a function argument, a loop target) counts as defined. That
can only make the proposal shorter, never wrong. A cell that does not parse
(for example cell magics other than ``%%time``) is skipped.

The proposal is shown on the Playground tab as the "Setup" card, which the
learner can edit. The runner runs whatever is saved at the start of every
session and after every restart (docs/PLAYGROUND.md, "Setup code").
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .notebook import joined

#: name -> (module, imported name or None for a module alias).
#:
#: ``("numpy", None)`` renders as ``import numpy as np``;
#: ``("sklearn.ensemble", "IsolationForest")`` as
#: ``from sklearn.ensemble import IsolationForest``. Only names in this table
#: are ever proposed. Keep each entry the convention a reader would expect.
KNOWN_IMPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    # Scientific Python, under their conventional aliases.
    "np": ("numpy", None),
    "pd": ("pandas", None),
    "plt": ("matplotlib.pyplot", None),
    "mpl": ("matplotlib", None),
    "sns": ("seaborn", None),
    "stats": ("scipy", "stats"),
    # Standard library modules notebooks often use bare.
    "math": ("math", None),
    "os": ("os", None),
    "re": ("re", None),
    "json": ("json", None),
    "random": ("random", None),
    "itertools": ("itertools", None),
    "warnings": ("warnings", None),
    # Notebook display helpers.
    "display": ("IPython.display", "display"),
    "HTML": ("IPython.display", "HTML"),
    "Markdown": ("IPython.display", "Markdown"),
    # scikit-learn estimators and helpers.
    "IsolationForest": ("sklearn.ensemble", "IsolationForest"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "LinearRegression": ("sklearn.linear_model", "LinearRegression"),
    "Ridge": ("sklearn.linear_model", "Ridge"),
    "Lasso": ("sklearn.linear_model", "Lasso"),
    "KMeans": ("sklearn.cluster", "KMeans"),
    "DBSCAN": ("sklearn.cluster", "DBSCAN"),
    "PCA": ("sklearn.decomposition", "PCA"),
    "LocalOutlierFactor": ("sklearn.neighbors", "LocalOutlierFactor"),
    "NearestNeighbors": ("sklearn.neighbors", "NearestNeighbors"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "StandardScaler": ("sklearn.preprocessing", "StandardScaler"),
    "MinMaxScaler": ("sklearn.preprocessing", "MinMaxScaler"),
    "RobustScaler": ("sklearn.preprocessing", "RobustScaler"),
    "OneHotEncoder": ("sklearn.preprocessing", "OneHotEncoder"),
    "LabelEncoder": ("sklearn.preprocessing", "LabelEncoder"),
    "SimpleImputer": ("sklearn.impute", "SimpleImputer"),
    "Pipeline": ("sklearn.pipeline", "Pipeline"),
    "make_pipeline": ("sklearn.pipeline", "make_pipeline"),
    "train_test_split": ("sklearn.model_selection", "train_test_split"),
    "cross_val_score": ("sklearn.model_selection", "cross_val_score"),
    "GridSearchCV": ("sklearn.model_selection", "GridSearchCV"),
    "KFold": ("sklearn.model_selection", "KFold"),
    "StratifiedKFold": ("sklearn.model_selection", "StratifiedKFold"),
    "accuracy_score": ("sklearn.metrics", "accuracy_score"),
    "precision_score": ("sklearn.metrics", "precision_score"),
    "recall_score": ("sklearn.metrics", "recall_score"),
    "f1_score": ("sklearn.metrics", "f1_score"),
    "roc_auc_score": ("sklearn.metrics", "roc_auc_score"),
    "confusion_matrix": ("sklearn.metrics", "confusion_matrix"),
    "classification_report": ("sklearn.metrics", "classification_report"),
    "mean_squared_error": ("sklearn.metrics", "mean_squared_error"),
}

#: Cell magics whose body is ordinary Python.
_PYTHON_CELL_MAGICS = {"%%time", "%%timeit", "%%capture", "%%prun"}
_ASSIGN_FROM_SHELL = re.compile(r"^(\s*)([A-Za-z_][\w\s,]*?)\s*=\s*[!%]")
_INDENT = re.compile(r"^[ \t]+")


def _python_source(source: str) -> Optional[str]:
    """The cell as plain Python: IPython-only lines become ``pass``, and a
    leading indent is stripped the way Jupyter did. None for a cell magic
    whose body is not Python."""
    lines = source.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines[0].split()[0]
        if magic not in _PYTHON_CELL_MAGICS:
            return None
        lines = lines[1:]
    match = _INDENT.match(lines[0]) if lines else None
    if match:
        space = match.group(0)
        lines = [line[len(space):] if line.startswith(space) else line for line in lines]
    out = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        assigned = _ASSIGN_FROM_SHELL.match(line)
        if assigned:
            out.append(f"{assigned.group(1)}{assigned.group(2).strip()} = None")
        elif stripped.startswith(("%", "!")):
            out.append(f"{indent}pass")
        else:
            out.append(line)
    return "\n".join(out)


def _bound_by_import(alias: ast.alias) -> str:
    if alias.asname:
        return alias.asname
    return alias.name.split(".")[0]


def _names(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    """(defined, used) names in one parsed cell."""
    defined: Set[str] = set()
    used: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (used if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update(_bound_by_import(a) for a in node.names if a.name != "*")
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            defined.add(node.rest)
    return defined, used


def undefined_names(sources: Iterable[str]) -> List[str]:
    """Names the code loads but never defines or imports in any cell, sorted.

    Builtins are included (``print``); the caller filters by its table.
    """
    defined: Set[str] = set()
    used: Set[str] = set()
    for source in sources:
        text = _python_source(source)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        d, u = _names(tree)
        defined |= d
        used |= u
    return sorted(used - defined)


def render_imports(names: Iterable[str]) -> str:
    """Import lines for the names in :data:`KNOWN_IMPORTS`, in table order:
    module imports first, then one ``from`` line per module."""
    wanted = set(names)
    plain: List[str] = []
    grouped: Dict[str, List[str]] = {}
    for name, (module, member) in KNOWN_IMPORTS.items():
        if name not in wanted:
            continue
        if member is None:
            plain.append(f"import {module}" if module == name else f"import {module} as {name}")
        else:
            grouped.setdefault(module, []).append(member if member == name else f"{member} as {name}")
    lines = plain + [f"from {module} import {', '.join(members)}" for module, members in grouped.items()]
    return "\n".join(lines) + ("\n" if lines else "")


def code_sources(nb: Dict[str, Any]) -> List[str]:
    return [
        joined(cell.get("source"))
        for cell in (nb or {}).get("cells") or []
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]


def suggest_setup(nb: Dict[str, Any]) -> str:
    """The proposed setup code for a notebook; empty when nothing is missing."""
    missing = undefined_names(code_sources(nb))
    return render_imports(n for n in missing if n in KNOWN_IMPORTS)
