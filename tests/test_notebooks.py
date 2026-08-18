"""The notebooks, checked for the things a notebook can be broken in.

Neither notebook's output is asserted here -- they draw pictures and play games
and both take minutes.  What is asserted is what breaks silently: a generator
that emits a cell which is not valid Python, and a first cell that no longer
fetches the repository, which is the difference between a notebook that runs on
Colab and one that fails on its first import with a message about a missing
module rather than a missing checkout.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = ("FifthFrontierWar.ipynb", "InvasionEarth.ipynb")

#: IPython line and cell magics, which are not Python and are not meant to be.
MAGIC = re.compile(r"^\s*[%!][A-Za-z]")


def cells(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def source(cell) -> str:
    return "".join(cell["source"])


class TestNotebooks(unittest.TestCase):
    def test_both_notebooks_are_present_and_are_notebooks(self):
        for name in NOTEBOOKS:
            found = cells(name)
            self.assertGreater(len(found), 10, name)
            self.assertTrue(all(c["cell_type"] in ("code", "markdown")
                                for c in found), name)

    def test_every_code_cell_is_valid_python(self):
        for name in NOTEBOOKS:
            for index, cell in enumerate(cells(name)):
                if cell["cell_type"] != "code":
                    continue
                text = "\n".join(line for line in source(cell).split("\n")
                                 if not MAGIC.match(line))
                try:
                    compile(text, "%s cell %d" % (name, index), "exec")
                except SyntaxError as exc:      # pragma: no cover - the point
                    self.fail("%s cell %d does not parse: %s"
                              % (name, index, exc))

    def test_the_first_code_cell_fetches_the_repository(self):
        """Without it the first import fails on Colab and says nothing useful."""
        from notebook_bootstrap import BRANCH, REPO
        for name in NOTEBOOKS:
            first = next(c for c in cells(name) if c["cell_type"] == "code")
            text = source(first)
            self.assertIn("repository_root", text, name)
            self.assertIn(REPO, text, name)
            self.assertIn(BRANCH, text, name)

    def test_the_bootstrap_does_not_clone_over_a_checkout(self):
        """Run from inside the repository it must do nothing but chdir."""
        from notebook_bootstrap import CODE
        env = {"__name__": "__main__"}
        here = os.getcwd()
        try:
            os.chdir(ROOT)
            exec(compile(CODE, "<bootstrap>", "exec"), env)
            self.assertEqual(os.path.realpath(os.getcwd()),
                             os.path.realpath(ROOT))
            self.assertFalse(os.path.exists(os.path.join(ROOT, "ffw-repo")))
        finally:
            os.chdir(here)

    def test_the_generators_reproduce_what_is_committed(self):
        """A notebook edited by hand rather than regenerated would drift."""
        import build_ie_notebook
        import build_notebook
        for module, name in ((build_notebook, "FifthFrontierWar.ipynb"),
                             (build_ie_notebook, "InvasionEarth.ipynb")):
            committed = cells(name)
            self.assertEqual(len(module.CELLS), len(committed), name)
            for (kind, text), cell in zip(module.CELLS, committed):
                self.assertEqual("markdown" if kind == "md" else "code",
                                 cell["cell_type"], name)
                self.assertEqual(text, source(cell), name)


if __name__ == "__main__":
    unittest.main()
