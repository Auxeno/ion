"""MkDocs build hooks for the Ion docs site."""

import pathlib

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
