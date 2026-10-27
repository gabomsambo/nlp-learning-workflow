"""The leading-indent rule the runner applies before a cell runs.

Jupyter (IPython <= 8) stripped the first line's indent from every line that
starts with it. The live-kernel versions of these checks are in
test_kernel_live.py.
"""

import pytest

from playground_runner.indent import install, leading_indent, strip_leading_indent


def test_an_indented_cell_is_dedented():
    assert strip_leading_indent("  x = 1\n  print(x)\n") == "x = 1\nprint(x)\n"


def test_unindented_trailing_comments_do_not_stop_it():
    code = "  y = 2\n  print(y)\n# a note\n#another\n"
    assert strip_leading_indent(code) == "y = 2\nprint(y)\n# a note\n#another\n"


def test_a_normally_indented_cell_is_unchanged():
    code = "for i in range(3):\n    print(i)\n"
    assert strip_leading_indent(code) == code


def test_a_mis_indented_cell_stays_mis_indented():
    # First line not indented: nothing to strip, so Python still raises.
    assert strip_leading_indent("x = 1\n  y = 2\n") == "x = 1\n  y = 2\n"
    # A later line indented less than the first keeps its indent.
    assert strip_leading_indent("    x = 1\n  y = 2\n") == "x = 1\n  y = 2\n"
    with pytest.raises(IndentationError):
        compile(strip_leading_indent("    x = 1\n  y = 2\n"), "<cell>", "exec")


def test_leading_blank_lines_are_skipped_first():
    assert strip_leading_indent("\n\n  a = 1\n  b = 2") == "a = 1\nb = 2"


def test_tabs_count_as_indent():
    assert leading_indent(["\tA\n", "\tB\n"]) == ["A\n", "B\n"]
    assert leading_indent([]) == []


def test_install_replaces_ipythons_own_transform():
    it2 = pytest.importorskip("IPython.core.inputtransformer2")

    class Shell:
        input_transformer_manager = it2.TransformerManager()

    shell = Shell()
    install(shell)
    transforms = shell.input_transformer_manager.cleanup_transforms
    assert leading_indent in transforms
    assert it2.leading_indent not in transforms
    install(shell)  # idempotent
    assert shell.input_transformer_manager.cleanup_transforms.count(leading_indent) == 1
    out = shell.input_transformer_manager.transform_cell("  y = 2\n  print(y)\n# note\n")
    assert out.startswith("y = 2\nprint(y)\n")
