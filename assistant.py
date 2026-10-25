"""What Note Detection says: short, clear sentences built around the total."""
from collections import Counter
import re


def describe(notes):
    """Spoken total for a list of note values, e.g. [1000, 1000, 50] -> "You have 2050 rupees." """
    if not notes:
        return "I don't see any notes."
    return f"You have {sum(notes)} rupees."


NUMBER_WORDS = ["", "a", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def name_notes(counts):
    """Counter({1000: 2, 50: 1}) -> ("two 1000 rupee notes and a 50 rupee note", 3)"""
    parts = [f"{NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else n} {value} rupee note{'s' if n > 1 else ''}"
             for value, n in sorted(counts.items(), reverse=True)]
    return " and ".join(parts), sum(counts.values())


def announce(before, after):
    """What to say when the notes in view change from `before` to `after`, or None to stay quiet."""
    if not after:
        return None
    total = f"You have {sum(after)} rupees."
    added, removed = Counter(after) - Counter(before), Counter(before) - Counter(after)
    if not before:
        if len(after) == 1:
            return f"Note detected. This is a {after[0]} rupee note."
        return f"{name_notes(added)[0].capitalize()} detected. {total}"
    sentences = []
    if added:
        names, count = name_notes(added)
        sentences.append(f"{names.capitalize()} {'has' if count == 1 else 'have'} been added.")
    if removed:
        names, count = name_notes(removed)
        sentences.append(f"{names.capitalize()} {'has' if count == 1 else 'have'} been removed.")
    return " ".join(sentences + [total])


INTENTS = [
    ("money", r"how much|total|money|amount|how many|count|which note|what note|what is this|what's this|rupee"),
    ("repeat", r"repeat|again|say that|didn't hear|what did you say"),
    ("thanks", r"thank|shukriya|jazak"),
    ("help", r"what can you do|help me|how does this work"),
]


def reply(text, notes, last_spoken):
    """Answer what the user said, or return None if it was not meant for us."""
    text = text.lower()
    intent = next((name for name, pattern in INTENTS if re.search(pattern, text)), None)
    if intent == "money":
        return describe(notes)
    if intent == "repeat":
        return last_spoken or describe(notes)
    if intent == "thanks":
        return "You're welcome."
    if intent == "help":
        return "Hold your notes in front of the camera and I'll tell you how much you have."
    return None
