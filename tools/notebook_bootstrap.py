"""The first cell of both notebooks: find the repository, wherever we are.

A notebook opened from a clone already has the packages next to it and needs
nothing.  A notebook opened on Colab arrives on its own -- the ``.ipynb`` is the
only file that travelled -- so ``import ffw`` fails on the first cell with a
``ModuleNotFoundError`` that says nothing about the actual problem, which is
that the repository is not there.

So the first cell fetches it.  The source lives here rather than in each
notebook generator because there are two notebooks and one bootstrap, and a
copy in each is a copy to forget to update.
"""

from __future__ import annotations

#: Where to clone from.  The repository is public, so no credentials are needed.
REPO = "https://github.com/pmcray/ffw.git"
BRANCH = "master"

MARKDOWN = """### First, the repository

This notebook needs the code it came from. If you are running it from a clone
it is already there and the cell below does nothing; on Colab, or anywhere else
the notebook has arrived on its own, it fetches the repository and moves into
it. Run it before anything else."""

CODE = '''# --- bootstrap -------------------------------------------------------------
import os
import subprocess
import sys

REPO = %r
BRANCH = %r
CHECKOUT = 'ffw-repo'        # deliberately not 'ffw': that is a package inside it


def repository_root():
    """The directory holding the packages, cloned first if it is not here."""
    here = os.path.abspath(os.getcwd())
    if os.path.isfile(os.path.join(here, 'ffw', '__init__.py')):
        return here                      # already inside a checkout
    checkout = os.path.join(here, CHECKOUT)
    if os.path.isfile(os.path.join(checkout, 'ffw', '__init__.py')):
        # a clone from an earlier run of this cell: bring it up to date, and
        # do not make a failure fatal -- an old copy still runs
        subprocess.run(['git', '-C', checkout, 'pull', '--ff-only', '--quiet'],
                       check=False)
        return checkout
    print('fetching %%s (branch %%s) ...' %% (REPO, BRANCH))
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH,
                    REPO, CHECKOUT], check=True)
    return checkout


root = repository_root()
os.chdir(root)
if root not in sys.path:
    sys.path.insert(0, root)

for package in ('numpy', 'matplotlib', 'ipywidgets'):
    try:
        __import__(package)
    except ImportError:
        print('installing %%s ...' %% package)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package],
                       check=False)

print('running in', os.getcwd())''' % (REPO, BRANCH)
