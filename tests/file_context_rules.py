"""The wording SPEC-ADDON-179/180/181 adds to a file question, one line per rule.

Each rule is the spec's sentence joined onto one line.
"""

from __future__ import annotations

S179 = (
    "- The File, Type and Folder lines at the start of the message, those that are "
    "there, describe the file: its name, its format and the folder it is in. They are "
    "part of what you were given. A question about the file itself (what it is, its "
    "name, its format, where it is) is answered from them; the rules below about "
    "passages, the image and frames are about what the file says and shows."
)
S180 = (
    "- If neither the passages nor the image contain the answer, say so plainly "
    "instead of guessing."
)
S181A = (
    "- Never call what you were given passages, frames or numbered items, in any "
    "language; speak of the file, what it says and what it shows."
)
S181B = (
    "- If the answer is not in what you were given, say so, but do not list other "
    "things you could not find out unless the question asks about them."
)
S181C = (
    "- Use square brackets only around a number you were given. Speak of a "
    "moment of a video as the video at its time."
)
PASSAGES_ONLY_SENTENCE = (
    "- If the passages do not contain the answer, say so plainly instead of guessing."
)
OLD_OPENER = (
    "You answer a question about a single file, using only the numbered passages below."
)
NEW_OPENER = (
    "You answer a question about a single file, using only the numbered passages "
    "below and the File, Type and Folder lines."
)
IMAGE_RULE = (
    "An image of the file is attached. Where the image and the passages differ, "
    "take exact characters from the passages."
)
ANSWER_LINE = "- Answer in short Markdown"
TOP_OF_DRIVE = "(top of the drive)"
LOFT_MIME = "application/vnd.litloft.loft+json"


def block(*, file: str | None, type_: str | None, folder: str | None = None) -> str:
    """The file block as it starts a user prompt; ``None`` leaves a line out."""
    lines = [
        f"{label}: {value}"
        for label, value in (("File", file), ("Type", type_), ("Folder", folder))
        if value is not None
    ]
    return "\n".join(lines) + "\n\n" if lines else ""
