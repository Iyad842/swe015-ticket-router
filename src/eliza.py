"""ELIZA-style rule-based replies for greetings and small talk (area B).

Each rule is a regular expression plus response templates. Templates may
use {0}, which is filled with the captured text after the classic ELIZA
pronoun reflection ("my" -> "your", "I am" -> "you are").
"""
from __future__ import annotations

import re

from .text_utils import tokenize, tr_lower

REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "mine": "yours", "am": "are",
    "i'm": "you're", "myself": "yourself", "you": "I", "your": "my",
    "yours": "mine", "are": "am",
}


def reflect(fragment: str) -> str:
    words = fragment.lower().strip(" .!?").split()
    return " ".join(REFLECTIONS.get(w, w) for w in words)


# (pattern, responses, is_small_talk)
# is_small_talk=True means a short message matching the rule is answered
# by the bot alone and is not routed to a team.
RULES: list[tuple[re.Pattern, list[str], bool]] = [
    (re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I),
     ["Hello! How can I help you with your appliance or account today?",
      "Hi there! Tell me what is wrong and I will route you to the right team."], True),
    (re.compile(r"^\s*(merhaba|selam|günaydın|iyi (günler|akşamlar))", re.I),
     ["Merhaba! Size nasıl yardımcı olabilirim?",
      "Selam! Sorununuzu yazın, sizi doğru ekibe yönlendireyim."], True),
    (re.compile(r"\bhow are you\b", re.I),
     ["I'm just a program, but I'm doing well. What can I do for you?"], True),
    (re.compile(r"\bnas[ıi]ls[ıi]n", re.I),
     ["Ben bir programım ama iyiyim, teşekkürler! Size nasıl yardımcı olabilirim?"], True),
    (re.compile(r"\b(thanks|thank you|thx)\b", re.I),
     ["You're welcome! Anything else I can help with?"], True),
    (re.compile(r"(teşekkür|tesekkur|sağ ?ol|sag ?ol)", re.I),
     ["Rica ederim! Başka bir konuda yardımcı olabilir miyim?"], True),
    (re.compile(r"\b(bye|goodbye|see you)\b", re.I),
     ["Goodbye! Have a nice day."], True),
    (re.compile(r"(görüşürüz|hoşça ?kal|hosca ?kal)", re.I),
     ["Görüşmek üzere, iyi günler!"], True),
    (re.compile(r"\bare you (a )?(bot|robot|human|real)\b", re.I),
     ["I'm an automated assistant. If you need a person, just ask for a human agent."], True),
    (re.compile(r"(bot|robot) mu ?sun", re.I),
     ["Ben otomatik bir asistanım. İsterseniz sizi bir müşteri temsilcisine bağlayabilirim."], True),
    # Classic ELIZA reflection rules
    (re.compile(r"\bi feel (.+)", re.I),
     ["Why do you feel {0}? Tell me what happened with your device."], False),
    (re.compile(r"\bi(?: am|'m) (.+)", re.I),
     ["How long have you been {0}? I'd like to help."], False),
    (re.compile(r"\bi need (.+)", re.I),
     ["Why do you need {0}? Let's see what I can do."], False),
]

FALLBACK_EN = "I see. Could you tell me a bit more so I can help?"
FALLBACK_TR = "Anlıyorum. Biraz daha detay verebilir misiniz?"

MAX_SMALL_TALK_TOKENS = 6


def _apply(rule_pat: re.Pattern, responses: list[str], text: str) -> str | None:
    m = rule_pat.search(text)
    if not m:
        return None
    template = responses[len(text) % len(responses)]
    groups = [g for g in m.groups() if g]
    return template.format(reflect(groups[0]) if groups else "")


def small_talk_reply(text: str) -> str | None:
    """Reply only if the message is *pure* small talk (short + small-talk rule).

    "hello" -> reply;  "hello my dishwasher is broken" -> None (route it).
    """
    if len(tokenize(tr_lower(text))) > MAX_SMALL_TALK_TOKENS:
        return None
    reply, rest = None, text
    for pat, responses, is_small_talk in RULES:
        if is_small_talk and pat.search(rest):
            reply = reply or _apply(pat, responses, text)
            rest = pat.sub(" ", rest)
    # what is left after removing the small-talk phrases must be (almost) empty
    if reply and len(tokenize(tr_lower(rest))) <= 2:
        return reply
    return None


def eliza_reply(text: str, lang: str = "en") -> str:
    """Always returns a reply (used when a model predicts the small_talk domain)."""
    for pat, responses, _ in RULES:
        reply = _apply(pat, responses, text)
        if reply:
            return reply
    return FALLBACK_TR if lang == "tr" else FALLBACK_EN
