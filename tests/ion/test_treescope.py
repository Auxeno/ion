"""Tests for treescope enable/disable in non-IPython environments."""

import ion


def test_enable_treescope_no_ipython():
    """Enable treescope is a no-op outside IPython."""
    ion.enable_treescope()


def test_enable_treescope_everything_no_ipython():
    """Enable treescope with everything=True is a no-op outside IPython."""
    ion.enable_treescope(everything=True)


def test_disable_treescope_no_ipython():
    """Disable treescope is a no-op outside IPython."""
    ion.disable_treescope()
