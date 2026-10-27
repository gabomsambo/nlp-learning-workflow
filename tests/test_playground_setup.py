"""Setup code: finding names a notebook uses but never wrote down, and the
store and import paths that keep it. Nothing here reaches a database."""

from unittest.mock import MagicMock, patch

import pytest

from nlp_pillars.playground import SETUP_MIGRATION, store
from nlp_pillars.playground.setup import (
    KNOWN_IMPORTS,
    render_imports,
    suggest_setup,
    undefined_names,
)


def _nb(*sources):
    return {"cells": [{"cell_type": "code", "source": s} for s in sources]}


# -- name detection ----------------------------------------------------------------


def test_a_name_used_but_never_defined_is_found():
    assert "np" in undefined_names(["x = np.array([1, 2])"])


def test_attribute_access_counts_as_using_the_name():
    missing = undefined_names(["idx = np.where(scores < 0)[0]", "plt.hist(scores)"])
    assert {"np", "plt"} <= set(missing)
    assert "where" not in missing and "hist" not in missing


def test_a_name_defined_in_an_earlier_cell_is_not_missing():
    missing = undefined_names(["import numpy as np\nscores = np.zeros(3)", "print(np.mean(scores))"])
    assert "np" not in missing and "scores" not in missing


def test_a_name_defined_anywhere_in_the_notebook_is_not_missing():
    # Never defined is the rule: a later definition still counts.
    assert "model" not in undefined_names(["model.fit(X)", "model = make()"])


def test_imported_under_another_alias_is_still_missing():
    missing = undefined_names(["import matplotlib.pyplot as pyplot\nplt.plot([1])", "import numpy\nnp.ones(2)"])
    assert {"plt", "np"} <= set(missing)
    assert "pyplot" not in missing and "numpy" not in missing


def test_definitions_of_every_kind_count():
    code = (
        "def f(a, *rest, k=1, **kw):\n    return a, rest, k, kw\n"
        "class C: pass\n"
        "for i in range(2): pass\n"
        "with open('x') as fh: pass\n"
        "try:\n    pass\nexcept ValueError as err:\n    print(err)\n"
        "ys = [y for y in range(3)]\n"
        "if (n := 3): pass\n"
        "from sklearn.ensemble import IsolationForest\n"
    )
    missing = undefined_names([code, "f(1); C(); i; fh; ys; n; IsolationForest()"])
    assert not {"f", "C", "i", "fh", "y", "ys", "n", "IsolationForest", "a", "rest", "k", "kw"} & set(missing)


def test_ipython_syntax_and_a_leading_indent_do_not_hide_a_cell():
    cells = [
        "%matplotlib inline\nfiles = !ls\nnp.ones(2)",
        "  df = pd.DataFrame()\n  df.head()\n# a note",
        "%%time\nm = IsolationForest()",
    ]
    assert {"np", "pd", "IsolationForest"} <= set(undefined_names(cells))
    assert "files" not in undefined_names(cells)


def test_a_cell_that_does_not_parse_is_skipped():
    assert undefined_names(["%%bash\necho $HOME", "def broken(:", "np.ones(1)"]) == ["np"]


# -- the proposal ------------------------------------------------------------------


def test_the_table_maps_the_conventional_imports():
    assert render_imports(["np"]) == "import numpy as np\n"
    assert render_imports(["math"]) == "import math\n"
    assert render_imports(["pd", "plt", "sns", "display", "IsolationForest"]) == (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from IPython.display import display\n"
        "from sklearn.ensemble import IsolationForest\n"
    )


def test_names_from_one_module_share_a_line():
    assert render_imports(["StandardScaler", "MinMaxScaler"]) == (
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n"
    )


def test_every_table_entry_is_a_valid_import():
    import ast

    for name in KNOWN_IMPORTS:
        line = render_imports([name])
        tree = ast.parse(line)
        bound = {a.asname or a.name.split(".")[0] for node in tree.body for a in node.names}
        assert bound == {name}, line


def test_only_names_in_the_table_are_proposed():
    nb = _nb(
        "scores = model.decision_function(X)\nidx = np.where(scores < 0)\nplt.plot(scores)",
        "display(df.head())\nm = IsolationForest(random_state=0)\nhelper(df)",
    )
    assert suggest_setup(nb) == (
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import display\n"
        "from sklearn.ensemble import IsolationForest\n"
    )


def test_a_notebook_that_imports_everything_needs_no_setup():
    nb = _nb("import numpy as np\nimport pandas as pd", "pd.Series(np.ones(2))")
    assert suggest_setup(nb) == ""


def test_markdown_cells_are_ignored():
    nb = {"cells": [{"cell_type": "markdown", "source": "Use np.where"}, {"cell_type": "code", "source": ["x = 1\n", "x"]}]}
    assert suggest_setup(nb) == ""


# -- store -------------------------------------------------------------------------


PG_ID = "6a1f6c1e-1111-4111-8111-111111111111"
PGRST204 = (
    "HTTP 400 {\"code\":\"PGRST204\",\"message\":\"Could not find the 'setup_code' column "
    "of 'playgrounds' in the schema cache\"}"
)


def _import_kwargs(**over):
    kw = dict(playground_id=PG_ID, paper_id="arxiv:1", title="T", notebook={"cells": []},
              notebook_name="t.ipynb", datasets=[], setup_code="import numpy as np\n")
    kw.update(over)
    return kw


def test_a_missing_setup_column_names_migration_030_not_028():
    exc = store.store_error(PGRST204, "Could not save")
    assert isinstance(exc, store.PlaygroundSetupColumnMissing)
    assert SETUP_MIGRATION in str(exc)
    assert "028" not in str(exc).split("(")[0]


def test_import_without_migration_030_saves_the_notebook_without_setup_code():
    seen = []

    def save_row(**kw):
        seen.append(kw["setup_code"])
        if kw["setup_code"] is not None:
            raise store.store_error(PGRST204, "Could not save the notebook")
        return {"id": kw["playground_id"]}

    with patch.object(store, "_save_import_row", side_effect=save_row):
        assert store.save_import(**_import_kwargs()) == {"id": PG_ID}
    assert seen == ["import numpy as np\n", None]


def test_import_stores_the_setup_code_in_the_row():
    client = MagicMock()
    client.base_url = "http://pg"
    client.headers = {}
    client.client.post.return_value = MagicMock(status_code=201, text="[]", json=lambda: [])
    with patch("nlp_pillars.db.get_client", return_value=client):
        store.save_import(**_import_kwargs())
    row = client.client.post.call_args.kwargs["json"][0]
    assert row["setup_code"] == "import numpy as np\n"


def test_save_setup_without_migration_030_names_it():
    client = MagicMock()
    error = Exception("400")
    error.response = MagicMock(status_code=400, text=PGRST204[9:])
    client.table.return_value.eq.return_value.update.side_effect = error
    with patch("nlp_pillars.db.get_client", return_value=client):
        with pytest.raises(store.PlaygroundSetupColumnMissing, match="030_playground_setup_code.sql"):
            store.save_setup(PG_ID, "import numpy as np\n")


def test_the_import_proposes_setup_code_for_the_new_notebook():
    import json

    from nlp_pillars.playground import service as pg_service

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "code", "source": "np.ones(3)", "metadata": {}, "outputs": [], "execution_count": None}]}
    cache = MagicMock()
    cache.cache.return_value = []
    with patch.object(store, "paper_row", return_value={"id": "arxiv:1"}), \
         patch.object(store, "get_for_paper", return_value=None), \
         patch.object(store, "save_import", side_effect=lambda **kw: dict(kw, id=kw["playground_id"])) as save, \
         patch.object(store, "save_datasets"):
        pg_service.import_notebook("arxiv:1", json.dumps(nb).encode(), "n.ipynb", cache=cache)
    assert save.call_args.kwargs["setup_code"] == "import numpy as np\n"
