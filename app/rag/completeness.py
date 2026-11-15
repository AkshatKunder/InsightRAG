"""Completeness checks for list and definition answers (Phase 3.1, Task 3).

Number checks cannot see omissions, so two generic question types are
checked against the retrieved evidence:

* **list**: "What are <owner>'s <plural noun>?". The evidence must contain an
  enumeration "<count?> <noun>: A, B, ... and Z". When the document states a
  count, it must equal the number of parsed items, otherwise the list is not
  trusted. The answer must name every item (normalised, "JVs" = "joint
  ventures") and must not present extra entries (a numbered or bulleted item
  that is not in the document's list).
* **definition**: "What is <Entity>'s primary/main/core business?" / "What does
  <Entity> do?". The evidence sentence "<Entity> is (one of the world's
  leading) <description> company/companies" defines the business. The answer
  must contain the description's core terms (participial modifiers such as
  "integrated" are optional, so concise answers are not rejected).

No evidence found -> ``None`` (not applicable): nothing is guessed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_WORD_NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_LIST_Q_RE = re.compile(
    r"^\s*(?:what|which)\s+(?:are|were)\s+(?:the\s+)?(?:(?P<owner>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)'s\s+)?"
    r"(?P<noun>[a-z][a-z -]*?s)\s*\??\s*$",
    re.I,
)
_DEF_Q_RE = re.compile(
    r"^\s*(?:what\s+is\s+(?P<a>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)'s\s+(?:primary|main|core|principal)\s+"
    r"(?:business|activity|activities)|what\s+does\s+(?P<b>[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*)\s+do)\b",
    re.I,
)
_DEF_STOP = {"one", "of", "the", "world", "worlds", "world's", "leading", "largest", "a", "an", "and", "global", "s"}


def _norm(text: str) -> str:
    text = re.sub(r"(?<![a-z])jvs?(?![a-z])", "joint ventures", text.lower().replace("&", " and "))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _sentences(chunks: Sequence[Mapping]):
    for chunk in chunks:
        text = " ".join(str(chunk.get("text", "")).split())
        for sentence in re.split(r"(?<=\.)\s+", text):
            yield chunk, sentence


def _split_items(listing: str) -> list[str]:
    listing = re.split(r",\s*(?:with|while|which|whereas)\b|;", listing)[0]
    parts = [p.strip(" .") for p in listing.split(",") if p.strip(" .")]
    if parts:
        last = re.split(r"\s+and\s+", parts[-1])
        if len(last) == 2:
            parts = parts[:-1] + last
    return [re.sub(r"\s*\(.*?\)", "", p).strip() for p in parts if p]


def list_evidence(question: str, chunks: Sequence[Mapping]) -> dict | None:
    match = _LIST_Q_RE.match(question)
    if not match:
        return None
    noun = " ".join(match.group("noun").split())
    pattern = re.compile(rf"(?:(?P<count>\b(?:\d+|{'|'.join(_WORD_NUMBERS)})\b)\s+(?:[\w-]+\s+){{0,3}})?{re.escape(noun)}[^:.]{{0,60}}:\s*(?P<list>[^.]+)", re.I)
    for chunk, sentence in _sentences(chunks):
        found = pattern.search(sentence)
        if not found:
            continue
        items = _split_items(found.group("list"))
        count_word = (found.group("count") or "").lower()
        count = int(count_word) if count_word.isdigit() else _WORD_NUMBERS.get(count_word)
        if len(items) < 2 or (count is not None and count != len(items)):
            continue  # unreliable parse: do not judge
        return {"kind": "list", "noun": noun, "items": items, "count": count, "page": chunk.get("page"),
                "evidence": found.group(0)[:300]}
    return None


def definition_evidence(question: str, chunks: Sequence[Mapping]) -> dict | None:
    match = _DEF_Q_RE.match(question)
    if not match:
        return None
    entity = match.group("a") or match.group("b")
    pattern = re.compile(rf"\b{re.escape(entity)}\s+is\s+(?P<desc>[^.;]{{3,160}}?)\s+(?:companies|company|group|producer)\b", re.I)
    for chunk, sentence in _sentences(chunks):
        found = pattern.search(sentence)
        if not found:
            continue
        words = re.findall(r"[a-z]+", found.group("desc").lower().replace("’", "'"))
        core = [w for w in words if w not in _DEF_STOP and not w.endswith("ed")]
        if core:
            return {"kind": "definition", "entity": entity, "terms": list(dict.fromkeys(core)), "page": chunk.get("page"),
                    "evidence": found.group(0)[:300]}
    return None


def _answer_items(answer: str) -> list[str]:
    """Items an answer presents as a numbered/bulleted list (one per line or "1. X 2. Y" inline)."""

    items = re.findall(r"(?:^|\s)(?:\d+[.)]|[-*•])\s+([^\n]+?)(?=\s+\d+[.)]\s|\n|$)", answer)
    return [i.strip(" .,;") for i in items if i.strip(" .,;")]


def check(question: str, answer: str, chunks: Sequence[Mapping]) -> dict | None:
    """Completeness of ``answer``; None when the question type or evidence does not apply."""

    norm_answer = f" {_norm(answer)} "
    evidence = list_evidence(question, chunks)
    if evidence:
        items = evidence["items"]
        missing = [i for i in items if f" {_norm(i)} " not in norm_answer]
        listed = _answer_items(answer)
        known = [_norm(i) for i in items]
        extra = [i for i in listed if not any(k and k in _norm(i) for k in known)]
        status = "complete" if not missing and not extra else "incomplete"
        return {**evidence, "status": status, "missing": missing, "extra": extra}
    evidence = definition_evidence(question, chunks)
    if evidence:
        missing = [t for t in evidence["terms"] if f" {t} " not in norm_answer]
        return {**evidence, "status": "complete" if not missing else "incomplete", "missing": missing, "extra": []}
    return None


def feedback(result: Mapping) -> str:
    if result["kind"] == "list":
        count = f"{result['count']} " if result.get("count") else ""
        parts = [f"The document lists {count}{result['noun']}: {', '.join(result['items'])}."]
        if result["missing"]:
            parts.append(f"The answer omitted: {', '.join(result['missing'])}.")
        if result["extra"]:
            parts.append(f"The answer listed entries that are not among them: {', '.join(result['extra'])}.")
        return " ".join(parts)
    return (f"The document describes it as follows: \"{result['evidence']}\". "
            f"The answer omitted: {', '.join(result['missing'])}.")
