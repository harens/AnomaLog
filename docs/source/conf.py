"""Sphinx configuration for the AnomaLog documentation site."""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AnomaLog"
author = "Haren Samarasinghe"
copyright_text = f"{datetime.now(timezone.utc).year}, {author}"
copyright = copyright_text  # noqa: A001

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxext.opengraph",
    "sphinx_design",
    "myst_parser",
]
autosummary_generate = True

myst_enable_extensions = ["colon_fence"]

ogp_site_url = "https://harens.github.io/AnomaLog/"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".md"]

copybutton_prompt_text = "$ "

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "AnomaLog"
html_theme = "furo"
# html_favicon = "./_static/icons/favicon.png"
html_theme_options = {
    # "light_logo": "icons/logo-light-mode.svg",
    # "dark_logo": "icons/logo-dark-mode.svg",
    # "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/harens/AnomaLog",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" '
                'd="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.'
                "4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-"
                "2.69-.94-0.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-0.01-.53 "
                "0.63-.01 1.08.58 1.23.82 0.72 1.21 1.87.87 2.33.66 0.07-.52 "
                "0.28-.87 0.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87 0.31-1.59 "
                "0.82-2.15-0.08-.2-0.36-1.02 0.08-2.12 0 0 0.67-.21 2.2.82 "
                "0.64-.18 1.32-.27 2-.27 0.68 0 1.36 0.09 2 0.27 1.53-1.04 "
                "2.2-.82 2.2-.82 0.44 1.1 0.16 1.92 0.08 2.12 0.51 0.56 0.82 "
                "1.27 0.82 2.15 0 3.07-1.87 3.75-3.65 3.95 0.29 0.25 0.54 0.73 "
                "0.54 1.48 0 1.07-0.01 1.93-0.01 2.2 0 0.21 0.15 0.46 0.55 0.38"
                'A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>'
                "</svg>"
            ),
            "class": "",
        },
    ],
}
html_static_path = ["_static"]
