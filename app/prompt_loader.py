"""Jinja2 prompt loader for the intelligence addon.

Single entry point used by every prompt builder in the addon. Keeps
the templates file-based so prompt changes show up as readable diffs
on PRs instead of buried in Python f-string concatenation.
"""

from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PROMPT_DIR = Path(__file__).parent / "prompts"


@cache
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_PROMPT_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        autoescape=False,
    )


def render(name: str, /, **vars: object) -> str:
    """Render a prompt template with the given variables.

    Example::

        render("summaries/short_long_system.jinja2",
               language_instruction=lang_line)
    """
    return _env().get_template(name).render(**vars)
