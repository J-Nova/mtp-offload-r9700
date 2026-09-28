# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IFEval verifiable-instruction graders, stdlib-only.

Adapted from google-research/instruction_following_eval (instructions.py +
instructions_registry.py), trimmed so it needs no third-party packages:

  * `absl.logging` calls dropped.
  * `langdetect` dropped: the two English-case checks test case only, and
    `language:response_language` is reported as unsupported (see UNSUPPORTED).
  * `nltk` dropped: `count_words` uses the same ``\\w+`` regex the upstream
    RegexpTokenizer uses; `count_sentences` uses the upstream regex splitter
    (`split_into_sentences`) instead of the punkt model; `capital_word_frequency`
    tokenizes on ``\\w[\\w'-]*`` instead of `nltk.word_tokenize`.

The instruction checkers that remain are byte-for-byte equivalent to upstream in
their pass/fail logic apart from those tokenizer substitutions, so the
instruction-level accuracy this module reports is comparable to IFEval's.

`grade(row, response)` returns (checked, passed) where `checked` is the number of
instruction ids it could grade and `passed` how many the response satisfied; a row
whose ids include an unsupported instruction is refused (`checked == 0`).
"""

from __future__ import annotations

import collections
import json
import re
import string

# ---------------------------------------------------------------------------
# instructions_util (ported)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")

_ALPHABETS = r"([A-Za-z])"
_PREFIXES = r"(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = r"(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|"
             r"Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)")
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + r"[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS,
                  lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + r"[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + r"[.]" + _ALPHABETS + r"[.]" + _ALPHABETS + r"[.]",
                  "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + r"[.]" + _ALPHABETS + r"[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + r"[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + r"[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + r"[.]", " \\1<prd>", text)
    if "\u201d" in text:
        text = text.replace(".\u201d", "\u201d.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_sentences(text: str) -> int:
    return len(split_into_sentences(text))


# ---------------------------------------------------------------------------
# instruction checkers (ported)
# ---------------------------------------------------------------------------

_COMPARISON_RELATION = ("less than", "at least")

_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.")
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")


def _relation(relation, actual, threshold):
    if relation == _COMPARISON_RELATION[0]:
        return actual < threshold
    return actual >= threshold


def _number_of_sentences(kwargs, value, _row):
    return _relation(kwargs.get("relation"), count_sentences(value),
                     kwargs.get("num_sentences"))


def _number_of_words(kwargs, value, _row):
    return _relation(kwargs.get("relation"), count_words(value), kwargs.get("num_words"))


def _number_of_paragraphs(kwargs, value, _row):
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                num -= 1
            else:
                return False
    return num == kwargs.get("num_paragraphs")


def _paragraph_first_word(kwargs, value, _row):
    num_paragraphs_want = kwargs.get("num_paragraphs")
    nth = kwargs.get("nth_paragraph")
    first_word_want = (kwargs.get("first_word") or "").lower()
    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            num_paragraphs -= 1
    if nth is None or nth > num_paragraphs:
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    punctuation = {".", ",", "?", "!", "'", '"'}
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first_word = ""
    for letter in word:
        if letter in punctuation:
            break
        first_word += letter.lower()
    return num_paragraphs == num_paragraphs_want and first_word == first_word_want


def _placeholders(kwargs, value, _row):
    placeholders = re.findall(r"\[.*?\]", value)
    return len(placeholders) >= kwargs.get("num_placeholders", 0)


def _postscript(kwargs, value, _row):
    value = value.lower()
    marker = kwargs.get("postscript_marker")
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + (marker or "").lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _number_bullet_lists(kwargs, value, _row):
    bullets = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    return len(bullets) + len(dashes) == kwargs.get("num_bullets")


def _constrained_response(_kwargs, value, _row):
    value = value.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSE_OPTIONS)


def _number_highlighted_sections(kwargs, value, _row):
    num = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", value):
        if highlight.strip("*").strip():
            num += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", value):
        if highlight.removeprefix("**").removesuffix("**").strip():
            num += 1
    return num >= kwargs.get("num_highlights", 0)


def _multiple_sections(kwargs, value, _row):
    splitter = kwargs.get("section_spliter") or ""
    pattern = r"\s?" + splitter + r"\s?\d+\s?"
    sections = re.split(pattern, value)
    return len(sections) - 1 >= kwargs.get("num_sections", 0)


def _json_format(_kwargs, value, _row):
    value = (value.strip()
             .removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
             .removeprefix("```").removesuffix("```").strip())
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _title(_kwargs, value, _row):
    for title in re.findall(r"<<[^\n]+>>", value):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def _two_responses(_kwargs, value, _row):
    valid = []
    responses = value.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index not in (0, len(responses) - 1):
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(kwargs, value, row):
    prompt = kwargs.get("prompt_to_repeat") or row.get("prompt") or ""
    return value.strip().lower().startswith(prompt.strip().lower())


def _end_checker(kwargs, value, _row):
    value = value.strip().strip('"').lower()
    end_phrase = (kwargs.get("end_phrase") or "").strip().lower()
    return value.endswith(end_phrase)


def _existence(kwargs, value, _row):
    for keyword in kwargs.get("keywords") or []:
        if not re.search(keyword, value, flags=re.IGNORECASE):
            return False
    return True


def _frequency(kwargs, value, _row):
    actual = len(re.findall(kwargs.get("keyword") or "", value, flags=re.IGNORECASE))
    return _relation(kwargs.get("relation"), actual, kwargs.get("frequency"))


def _forbidden_words(kwargs, value, _row):
    for word in kwargs.get("forbidden_words") or []:
        if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
            return False
    return True


def _letter_frequency(kwargs, value, _row):
    counts = collections.Counter(value.lower())
    letter = (kwargs.get("letter") or "").lower()
    relation = kwargs.get("let_relation")
    return _relation(relation, counts[letter], kwargs.get("let_frequency"))


def _capital_word_frequency(kwargs, value, _row):
    words = re.findall(r"\w[\w'-]*", value)
    capital_words = len([w for w in words if w.isupper()])
    return _relation(kwargs.get("capital_relation"), capital_words,
                     kwargs.get("capital_frequency"))


def _english_capital(_kwargs, value, _row):
    return value.isupper()


def _english_lowercase(_kwargs, value, _row):
    return value.islower()


def _no_comma(_kwargs, value, _row):
    return not re.search(r"\,", value)


def _quotation(_kwargs, value, _row):
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


# instruction id -> (checker, kwargs keys it reads). Order mirrors upstream.
CHECKERS = {
    "keywords:existence": (_existence, ("keywords",)),
    "keywords:frequency": (_frequency, ("keyword", "frequency", "relation")),
    "keywords:forbidden_words": (_forbidden_words, ("forbidden_words",)),
    "keywords:letter_frequency": (_letter_frequency, ("letter", "let_frequency", "let_relation")),
    "length_constraints:number_sentences": (_number_of_sentences, ("num_sentences", "relation")),
    "length_constraints:number_paragraphs": (_number_of_paragraphs, ("num_paragraphs",)),
    "length_constraints:number_words": (_number_of_words, ("num_words", "relation")),
    "length_constraints:nth_paragraph_first_word": (
        _paragraph_first_word, ("num_paragraphs", "nth_paragraph", "first_word")),
    "detectable_content:number_placeholders": (_placeholders, ("num_placeholders",)),
    "detectable_content:postscript": (_postscript, ("postscript_marker",)),
    "detectable_format:number_bullet_lists": (_number_bullet_lists, ("num_bullets",)),
    "detectable_format:constrained_response": (_constrained_response, ()),
    "detectable_format:number_highlighted_sections": (
        _number_highlighted_sections, ("num_highlights",)),
    "detectable_format:multiple_sections": (_multiple_sections, ("section_spliter", "num_sections")),
    "detectable_format:json_format": (_json_format, ()),
    "detectable_format:title": (_title, ()),
    "combination:two_responses": (_two_responses, ()),
    "combination:repeat_prompt": (_repeat_prompt, ("prompt_to_repeat",)),
    "startend:end_checker": (_end_checker, ("end_phrase",)),
    "change_case:capital_word_frequency": (
        _capital_word_frequency, ("capital_frequency", "capital_relation")),
    "change_case:english_capital": (_english_capital, ()),
    "change_case:english_lowercase": (_english_lowercase, ()),
    "punctuation:no_comma": (_no_comma, ()),
    "startend:quotation": (_quotation, ()),
}

# Needs langdetect upstream; refused here so a row is skipped rather than
# mis-graded. 4 of the 541 IFEval prompts carry it.
UNSUPPORTED = {"language:response_language"}


def _kwargs_for(kwargs_list, index, keys):
    """Pull the named keys out of the kwargs entry, ignoring absent/None ones."""
    raw = kwargs_list[index] if index < len(kwargs_list) else {}
    if not isinstance(raw, dict):
        return {}
    return {k: raw[k] for k in keys if raw.get(k) is not None}


def grade(row: dict, response: str) -> tuple[int, int]:
    """Return (instructions_checked, instructions_passed) for one IFEval row.

    row: a fetched record with extra.instruction_id_list and extra.kwargs.
    checked == 0 means the row carries an unsupported instruction and was skipped.
    """
    extra = row.get("extra") or {}
    ids = extra.get("instruction_id_list") or []
    kwargs_list = extra.get("kwargs") or []
    if not ids or any(i in UNSUPPORTED for i in ids):
        return 0, 0
    checked = passed = 0
    for index, instruction_id in enumerate(ids):
        entry = CHECKERS.get(instruction_id)
        if entry is None:
            return 0, 0
        checker, keys = entry
        try:
            ok = bool(checker(_kwargs_for(kwargs_list, index, keys), response, row))
        except Exception:  # noqa: BLE001 - a malformed kwarg must not kill the run
            return 0, 0
        checked += 1
        passed += 1 if ok else 0
    return checked, passed
