"""MkDocs build hooks for the Ion docs site."""

import pathlib
import re

from pygments.lexers.python import PythonLexer
from pygments.token import Name, Operator, Punctuation

# Highlight calls and dotted names consistently in fenced and notebook code.
# The IPython lexer used by nbconvert inherits this Python lexer.


def on_config(config, **kwargs):
    if getattr(PythonLexer, "_ion_call_sites", False):
        return
    setattr(PythonLexer, "_ion_call_sites", True)
    tokenize = PythonLexer.get_tokens_unprocessed

    def highlight_call_sites(self, text, *args):
        pending = None
        after_dot = False
        for index, token, value in tokenize(self, text, *args):
            if pending is not None:
                if token is Punctuation and value == "(":
                    yield pending[0], Name.Function, pending[1]
                elif pending[2] or (token is Operator and value == "."):
                    yield pending[0], Name.Attribute, pending[1]
                else:
                    yield pending[0], Name, pending[1]
                pending = None
            if token is Name:
                pending = (index, value, after_dot)
            else:
                yield index, token, value
            after_dot = token is Operator and value == "."
        if pending is not None:
            yield pending[0], Name.Attribute if pending[2] else Name, pending[1]

    setattr(PythonLexer, "get_tokens_unprocessed", highlight_call_sites)


# Show Material's desktop navigation from 60em. Both values occur in the
# compiled CSS and JavaScript media queries.
BREAKPOINTS = {
    "76.234375em": "59.984375em",
    "76.25em": "60em",
}


# nbconvert and Plotly's notebook renderer inject MathJax 2 scripts into
# exported notebook HTML. The site uses MathJax 3, and loading both versions
# breaks typesetting on notebook pages.
MATHJAX_2_SCRIPTS = (
    re.compile(r'<script src="">\s*</script>'),
    re.compile(r'<script type="text/x-mathjax-config">.*?</script>', re.DOTALL),
    re.compile(
        r'<script src="https://cdnjs\.cloudflare\.com/ajax/libs/mathjax/2\.[^"]*">'
        r"</script>"
    ),
)


def on_post_page(output, page, config, **kwargs):
    """Remove notebook-exported MathJax 2 so the site's MathJax 3 can run."""
    if page.file.src_path.endswith(".ipynb"):
        for pattern in MATHJAX_2_SCRIPTS:
            output = pattern.sub("", output)
    return output


def on_post_build(config, **kwargs):
    site = pathlib.Path(config["site_dir"])
    assets = [
        *site.glob("assets/stylesheets/main.*.min.css"),
        *site.glob("assets/javascripts/bundle.*.min.js"),
    ]
    for path in assets:
        text = path.read_text()
        for old, new in BREAKPOINTS.items():
            text = text.replace(old, new)
        path.write_text(text)
