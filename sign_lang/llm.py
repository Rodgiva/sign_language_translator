"""Generative stage: ASL glosses -> English sentence.

Goes through the local Ollama server. No extra dependency needed: urllib from
the standard library is enough.

Quick test:
    python -c "from sign_lang.llm import to_sentence; print(to_sentence([[('I',0.9)],[('EAT',0.8)],[('APPLE',0.7)]]))"
"""

import json
import re
import urllib.request

HOST = "http://localhost:11434"
MODEL = "qwen3.5:4b"
TIMEOUT = 60  # the first call pays for loading the model
TEMPERATURE = 0.3
MAX_TOKENS = 80

SYSTEM = """You convert ASL (American Sign Language) glosses into natural English.

ASL glosses are uppercase word labels produced by a sign recognizer. ASL grammar
differs from English: it has no articles, no copula (to be), time markers come
first, and it often uses topic-comment order. Your job is to recover the English
sentence the signer meant.

Each numbered line is ONE sign, listed with the recognizer's three best guesses
and their confidence. The three are competing hypotheses for that single sign,
not three separate words: choose exactly one per line and discard the other two.
A sentence of four lines therefore uses exactly four of the listed words.

The first candidate is right about two thirds of the time, so take it by default;
fall back to a lower-ranked one only when the first makes the sentence incoherent.
Never replace a sign with a content word that appears in none of its candidates.

That constraint is about which sign you pick, nothing more. You must still supply
everything English needs and ASL leaves out: articles, forms of "to be",
prepositions, pronoun case, plurals, verb tense and inflection. Write a fluent
sentence in normal capitalisation, never a word-for-word transcription.

Glosses may carry a variant digit, as in PIZZA1 and CHRISTMAS2: two distinct
signs for the same English word. Drop the digit.

Reply with the English sentence only. No quotes, no explanation, no alternatives."""

_THINK = re.compile(r"<think>.*?</think>", re.S)


def format_signs(signs):
    """signs: one entry per sign, each one [(gloss, confidence), ...]."""
    lines = [
        f"{i + 1}. " + "  |  ".join(f"{g} ({c * 100:.0f}%)" for g, c in cands)
        for i, cands in enumerate(signs)
    ]
    return "Signs in order:\n" + "\n".join(lines)


def to_sentence(signs, model=MODEL, host=HOST, timeout=TIMEOUT):
    """Return the English sentence. Raises if Ollama does not answer."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_signs(signs)},
        ],
        "stream": False,
        "think": False,  # qwen3.x: keep the reasoning out of the reply
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())

    text = _THINK.sub("", body["message"]["content"]).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip('"').strip() if lines else ""
