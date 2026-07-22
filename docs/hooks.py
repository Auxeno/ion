"""MkDocs build hooks for the Ion docs site."""

import pathlib

from pygments.lexers.python import PythonLexer
from pygments.token import Name, Operator, Punctuation

# Pygments only tags Name.Function at def sites, so call-heavy example code
# renders almost entirely in the default text colour. Wrap the Python lexer
# with a one-token lookahead that retags a plain Name directly followed by an
# opening paren as Name.Function (nn.MLP(...), jax.grad(...)) and any other
# dotted name as Name.Attribute (optax.chain, model.at); extra.css colours
# both with the function blue. Arguments, kwargs, and bare variables stay
# plain. nbconvert's IPython lexer subclasses PythonLexer, so notebook pages
# inherit the retag. Build-time only; if a Pygments upgrade ever breaks the
# wrap, everything reverts to plain names.


def on_config(config, **kwargs):
    if getattr(PythonLexer, "_ion_call_sites", False):
        return
    PythonLexer._ion_call_sites = True
    inner = PythonLexer.get_tokens_unprocessed

    def with_call_sites(self, text, *args):
        pending = None
        after_dot = False
        for index, token, value in inner(self, text, *args):
            if pending is not None:
                if token is Punctuation and value == "(":
                    yield pending[0], Name.Function, pending[1]
                else:
                    yield pending[0], Name.Attribute if pending[2] else Name, pending[1]
                pending = None
            if token is Name:
                pending = (index, value, after_dot)
            else:
                yield index, token, value
            after_dot = token is Operator and value == "."
        if pending is not None:
            yield pending[0], Name.Attribute if pending[2] else Name, pending[1]

    PythonLexer.get_tokens_unprocessed = with_call_sites

# Material reveals the left navigation only from 76.25em, which on a laptop is
# roughly a full-width window. Shift that breakpoint, and its max-width drawer
# companion, down to 60em (960px) so the sidebar appears around 65% width. The
# value lives in both the compiled CSS and the JS media watcher, so we rewrite
# both to keep them in sync. Breakpoints below 60em (mobile, tablet) are left
# untouched, so small-screen layout is unchanged.
BREAKPOINTS = {
    "76.234375em": "59.984375em",  # max-width drawer companion (60em - 0.015625em)
    "76.25em": "60em",             # min-width desktop breakpoint
}


def on_post_build(config, **kwargs):
    site = pathlib.Path(config["site_dir"])
    assets = [*site.glob("assets/stylesheets/main.*.min.css"), *site.glob("assets/javascripts/bundle.*.min.js")]
    for path in assets:
        text = path.read_text()
        for old, new in BREAKPOINTS.items():
            text = text.replace(old, new)
        path.write_text(text)
