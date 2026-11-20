"""Classical NLP preprocessing (technique area B).

* Regular expressions: URL/e-mail masking, product model-code extraction
  (e.g. "DW-6534"), appliance error-code extraction (e.g. "E21"),
  repeated-letter squeezing ("pleaseee" -> "pleasee").
* Turkish-aware lower-casing and tokenization.
* Minimum edit distance (Levenshtein, dynamic programming) to correct
  misspelled product names against a product lexicon, e.g.
  "dishwaher" -> "dishwasher", "bulasik" -> "bulaşık".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Regular expressions
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Product model codes: "DW-6534", "wm-8020", "RF 450X", "WM8020".
# Lower-case letters are only accepted together with a hyphen so that normal
# phrases such as "in 2024" are not treated as model codes.
MODEL_CODE_RE = re.compile(
    r"\b(?:[A-Za-z]{1,3}-\d{3,5}[A-Za-z]{0,2}|[A-Z]{2,3}\s?\d{3,5}[A-Z]{0,2})\b"
)
# Appliance error codes: "E21", "F05", "E-07", "H1".
ERROR_CODE_RE = re.compile(r"\b[EFHefh]-?\d{1,3}\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

MODEL_PLACEHOLDER = "MODELCODE"

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_STOPWORDS = {"ve", "bir", "bu", "için", "mi", "mı", "mu", "mü", "ne", "değil",
                "çok", "var", "yok", "ile", "ama", "gibi", "lazım", "istiyorum"}
# Fold Turkish letters to ASCII, used only for *comparing* words.
_FOLD = str.maketrans("şŞıİğĞüÜöÖçÇ", "sSiIgGuUoOcC")

# Product lexicon (English + Turkish) used for spelling correction.
PRODUCT_LEXICON = [
    "dishwasher", "dishwashers", "washer", "washing", "machine", "dryer",
    "refrigerator", "fridge", "freezer", "oven", "microwave", "vacuum",
    "cleaner", "conditioner", "cooktop", "kettle", "filter", "drain", "hose",
    "bulaşık", "çamaşır", "makine", "makinesi", "makinem", "makinemin",
    "makinesinin", "buzdolabı", "buzdolabım", "buzdolabının", "fırın", "fırınım",
    "fırının", "kurutma", "klima", "klimam", "süpürge", "süpürgem", "mikrodalga",
    "ocak", "filtre", "hortum", "hortumu", "tahliye",
]


def tr_lower(text: str) -> str:
    """Lower-case text; 'İ' -> 'i' explicitly (Python would give 'i̇')."""
    return text.replace("İ", "i").lower()


def fold(text: str) -> str:
    return text.translate(_FOLD)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detect_language(text: str) -> str:
    if any(c in TR_CHARS for c in text):
        return "tr"
    toks = set(tokenize(tr_lower(text)))
    return "tr" if toks & TR_STOPWORDS else "en"


# --------------------------------------------------------------------------
# Minimum edit distance
# --------------------------------------------------------------------------
def min_edit_distance(a: str, b: str, sub_cost: int = 1) -> int:
    """Levenshtein distance with a two-row dynamic-programming table.

    D[i][j] = min(D[i-1][j] + 1        (deletion),
                  D[i][j-1] + 1        (insertion),
                  D[i-1][j-1] + c(i,j) (substitution, 0 if letters match))
    """
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else sub_cost
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


class SpellCorrector:
    """Corrects unknown tokens to the closest product-lexicon word.

    A token is only corrected if it is NOT a known word of the training
    vocabulary (so 'bridge' is never turned into 'fridge') and its edit
    distance (after folding Turkish letters) is small: <=1 for words of
    length <=6, <=2 for longer words. Tokens shorter than 4 letters are kept.
    """

    def __init__(self, vocab: set[str] | None = None, lexicon: list[str] | None = None):
        self.vocab = vocab or set()
        self.lexicon = lexicon or PRODUCT_LEXICON
        self.lex_set = set(self.lexicon)
        self.folded = [(fold(w), w) for w in self.lexicon]

    def correct(self, token: str) -> str | None:
        if len(token) < 4 or token in self.lex_set or token in self.vocab:
            return None
        ft = fold(token)
        max_d = 1 if len(token) <= 6 else 2
        best, best_d = None, max_d + 1
        for fw, w in self.folded:
            if abs(len(fw) - len(ft)) > max_d:
                continue
            d = min_edit_distance(ft, fw)
            if d < best_d:
                best, best_d = w, d
        return best if best is not None and best_d <= max_d else None


# --------------------------------------------------------------------------
# Full normalization pipeline
# --------------------------------------------------------------------------
class Normalizer:
    def __init__(self, vocab: set[str] | None = None):
        self.corrector = SpellCorrector(vocab)

    @classmethod
    def from_file(cls, path: str | Path) -> "Normalizer":
        path = Path(path)
        vocab = set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()
        return cls(vocab)

    def __call__(self, text: str) -> dict:
        raw = text
        text = URL_RE.sub(" url ", text)
        text = EMAIL_RE.sub(" email ", text)
        model_codes = [m.group(0) for m in MODEL_CODE_RE.finditer(text)]
        text = MODEL_CODE_RE.sub(f" {MODEL_PLACEHOLDER} ", text)
        error_codes = [m.group(0).upper() for m in ERROR_CODE_RE.finditer(text)]
        text = REPEAT_RE.sub(r"\1\1", text)
        text = SPACE_RE.sub(" ", text).strip()

        # spelling correction of product names (token level, keeps punctuation)
        corrections: list[tuple[str, str]] = []

        def _fix(match: re.Match) -> str:
            tok = match.group(0)
            if tok == MODEL_PLACEHOLDER:
                return tok
            fixed = self.corrector.correct(tr_lower(tok))
            if fixed:
                corrections.append((tok, fixed))
                return fixed
            return tok

        clean = TOKEN_RE.sub(_fix, text)
        lower = " ".join(tokenize(tr_lower(clean)))
        return {
            "raw": raw,
            "clean": clean,          # cased text for the Transformers
            "lower": lower,          # lower-cased tokens for tf-idf / embeddings
            "model_codes": model_codes,
            "error_codes": error_codes,
            "corrections": corrections,
            "lang": detect_language(raw),
        }
