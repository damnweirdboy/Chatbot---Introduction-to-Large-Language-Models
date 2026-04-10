"""
NLP Chatbot — SWE015: Introduction to Large Language Models
============================================================
Techniques used:
  1. Text Normalization     — lowercasing, punctuation removal
  2. Stemming               — Porter stemmer (text normalization)
  3. Regular Expressions    — Pattern-based ELIZA-style intent rules
  4. Edit Distance          — Levenshtein distance for typo correction
  5. N-gram TF-IDF          — unigrams + bigrams for semantic similarity
  6. Cosine Similarity      — ranking best-matching FAQ answers
  7. Intent Detection       — rule-based classifier on top of TF-IDF
  8. Confidence Threshold   — fallback when similarity is too low
"""

import json
import re
import os
import sys
import math
from collections import Counter

# ── Force unbuffered output so VS Code terminal shows responses immediately ──
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────
# 1. EDIT DISTANCE  (Levenshtein)
# ──────────────────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    """Return minimum edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ──────────────────────────────────────────────
# 2. TEXT NORMALISATION
# ──────────────────────────────────────────────
# Minimal Porter-style suffix stripping (stemmer)
SUFFIXES = ["ing", "tion", "ions", "ness", "ment", "ers", "ed", "es", "er", "ly", "s"]

def stem(word: str) -> str:
    """Simple suffix-stripping stemmer."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, stem each token."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stem(t) for t in text.split()]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# 3. SIMPLE TF-IDF (with N-gram support)
# ──────────────────────────────────────────────
def get_ngrams(tokens: list, n: int) -> list:
    """Return all n-grams from a token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def extract_features(text: str, ngram_range=(1, 2)) -> list:
    """Extract unigram + bigram features from normalised text."""
    tokens = text.split()
    features = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        features.extend(get_ngrams(tokens, n))
    return features

class TFIDFVectorizer:
    """
    Scratch TF-IDF vectorizer supporting N-grams.
    Demonstrates foundational NLP concepts without sklearn.
    """
    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range
        self.vocab: dict = {}
        self.idf: dict = {}

    def fit(self, corpus: list):
        # Build vocabulary
        doc_freq: Counter = Counter()
        for doc in corpus:
            feats = set(extract_features(doc, self.ngram_range))
            doc_freq.update(feats)
            for f in feats:
                if f not in self.vocab:
                    self.vocab[f] = len(self.vocab)
        # Compute IDF
        N = len(corpus)
        for term, df in doc_freq.items():
            self.idf[term] = math.log((N + 1) / (df + 1)) + 1  # smooth IDF

    def transform(self, corpus: list) -> list:
        """Return list of sparse dicts {term_idx: tf-idf}."""
        vectors = []
        for doc in corpus:
            feats = extract_features(doc, self.ngram_range)
            tf: Counter = Counter(feats)
            vec = {}
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf_score = count / len(feats) if feats else 0
                    vec[idx] = tf_score * self.idf.get(term, 0)
            vectors.append(vec)
        return vectors

    def fit_transform(self, corpus: list) -> list:
        self.fit(corpus)
        return self.transform(corpus)


def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(vec_a.get(k, 0) * v for k, v in vec_b.items())
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────
# 4. INTENT DETECTION (Regex / Rule-Based)
# ──────────────────────────────────────────────
INTENT_PATTERNS = [
    ("greet",        r"\b(hello|hi|hey|howdy|sup|yo|good\s*(morning|afternoon|evening|night)|whats\s*up|greetings)\b"),
    ("farewell",     r"\b(bye|goodbye|see\s*you|take\s*care|later|cya|quit|exit|farewell)\b"),
    ("thanks",       r"\b(thank(s|\s*you)?|appreciate|thx|ty|cheers)\b"),
    ("bot_info",     r"\b(who\s*are\s*you|your\s*name|what\s*are\s*you|how\s*do\s*you\s*work|are\s*you\s*a\s*bot|help|what\s*can\s*(you|i))\b"),
    ("schedule",     r"\b(when|what\s*time|what\s*day|schedule|start|end|duration|meet|class\s*time)\b"),
    ("course_info",  r"\b(instructor|professor|teacher|email|contact|course\s*(name|code|title|number))\b"),
    ("nlp_concepts", r"\b(nlp|bert|gpt|transformer|tfidf|tf.idf|word2vec|embed|token|stem|lemma|ngram|n.gram|attention|llm|language\s*model|edit\s*distance|cosine|ppmi|eliza|chatbot)\b"),
    ("assessment",   r"\b(assignment|project|grade|exam|midterm|final|homework|quiz|graded|grading)\b"),
    ("university_info", r"\b(university|school|campus|department|istinye|where|location|city)\b"),
]

def detect_intent(text: str) -> str:
    """Return intent label via regex pattern matching."""
    lower = text.lower()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, lower):
            return intent
    return "general"


# ──────────────────────────────────────────────
# 5. ELIZA-STYLE RESPONSE TEMPLATES
# ──────────────────────────────────────────────
ELIZA_RULES = [
    (r"i am (.*)",           "Why do you say you are {0}?"),
    (r"i feel (.*)",         "Tell me more about feeling {0}."),
    (r"i need (.*)",         "Why do you need {0}?"),
    (r"i want (.*)",         "What would it mean to you if you got {0}?"),
    (r"why (.*)\?",          "Why do you think {0}?"),
    (r"can you (.*)\?",      "Do you think I can {0}?"),
    (r"do you (.*)\?",       "I don't have feelings, but I can try to answer: {0}?"),
    (r"are you (.*)\?",      "Why does it matter whether I am {0}?"),
    (r"what is (.*)\?",      "Let me look up what {0} means for you..."),
    (r"tell me about (.*)",  "I'll try to explain {0}. Ask a specific question for better results!"),
]

def eliza_response(text: str):
    """Try ELIZA-style reflection. Returns string or None."""
    for pattern, template in ELIZA_RULES:
        m = re.match(pattern, text.strip().lower())
        if m:
            captured = m.group(1).strip()
            return template.format(captured)
    return None


# ──────────────────────────────────────────────
# 6. TYPO CORRECTION via Edit Distance
# ──────────────────────────────────────────────
def word_level_similarity(input_tokens: list, question_tokens: list) -> float:
    """
    Compare token lists word-by-word using edit distance.
    Returns a similarity score in [0, 1].
    Handles different length inputs robustly.
    """
    if not input_tokens or not question_tokens:
        return 0.0
    matched = 0
    for w in input_tokens:
        # find best-matching word in question
        best = min(levenshtein(w, qw) for qw in question_tokens)
        max_len = max(len(w), max(len(qw) for qw in question_tokens))
        if max_len == 0:
            continue
        if best / max_len <= 0.4:   # allow ~40% character edits per word
            matched += 1
    return matched / max(len(input_tokens), len(question_tokens))


def find_closest_question(user_input: str, questions: list, threshold: float = 0.45):
    """
    Find the most similar question using word-level edit distance.
    threshold: minimum word-level similarity score to accept (0‒1).
    Returns (index, score) or (None, 0).
    """
    input_tokens = user_input.split()
    best_idx, best_score = None, 0.0
    for i, q in enumerate(questions):
        q_tokens = q.split()
        score = word_level_similarity(input_tokens, q_tokens)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= threshold:
        return best_idx, best_score
    return None, 0.0


# ──────────────────────────────────────────────
# 7. LOAD DATA
# ──────────────────────────────────────────────
# Always resolve data.json relative to THIS script file,
# regardless of where VS Code / the terminal sets the working directory.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
try:
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"[ERROR] Cannot find data.json at: {DATA_FILE}")
    print("Make sure data.json is in the same folder")
    raise SystemExit(1)
except json.JSONDecodeError as e:
    print(f"[ERROR] data.json is not valid JSON: {e}")
    raise SystemExit(1)

raw_questions = [item["question"] for item in data]
answers       = [item["answer"]   for item in data]
intents       = [item.get("intent", "general") for item in data]

# Normalise questions for vectorization (keeps original for edit-distance)
norm_questions = [normalize(q) for q in raw_questions]

# Build TF-IDF index
vectorizer = TFIDFVectorizer(ngram_range=(1, 2))
doc_vectors = vectorizer.fit_transform(norm_questions)


# ──────────────────────────────────────────────
# 8. RETRIEVAL FUNCTION
# ──────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.08  # below this → try edit distance or fallback

def get_response(user_input: str) -> tuple:
    """
    Full retrieval pipeline:
      1. Detect intent (regex)
      2. Try ELIZA reflection
      3. TF-IDF cosine similarity retrieval
      4. Word-level edit-distance fallback for typos / short inputs
      5. Confidence threshold with friendly fallback
    Returns (answer, intent, confidence, method)
    """
    intent  = detect_intent(user_input)
    norm_in = normalize(user_input)

    # ── ELIZA reflection (for personal/free-form statements)
    eliza = eliza_response(user_input)

    # ── TF-IDF similarity ──
    query_vec = vectorizer.transform([norm_in])[0]
    similarities = [cosine_sim(query_vec, dv) for dv in doc_vectors]

    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    best_sim = similarities[best_idx]
    method = "tfidf-cosine"
    confidence = best_sim

    # ── Word-level edit-distance fallback (handles typos like "wihch instrucor") ──
    if best_sim < SIMILARITY_THRESHOLD:
        ed_idx, ed_score = find_closest_question(norm_in, norm_questions, threshold=0.45)
        if ed_idx is not None:
            best_idx  = ed_idx
            method    = f"word-edit-distance (score={ed_score:.2f})"
            confidence = ed_score
        else:
            method     = "no-match"
            confidence = 0.0

    # ── If ELIZA matched AND similarity is low → prefer ELIZA ──
    if eliza and confidence < 0.25:
        return eliza, intent, confidence, "eliza-reflection"

    # ── Final threshold check ──
    if confidence < SIMILARITY_THRESHOLD:
        fallback = (
            "I'm not sure I understand. Could you rephrase? "
            "You can ask about the course, schedule, NLP topics, or type 'help'."
        )
        return fallback, intent, confidence, "fallback"

    return answers[best_idx], intent, confidence, method


# ──────────────────────────────────────────────
# 9. DISPLAY HELPERS
# ──────────────────────────────────────────────
DIVIDER = "─" * 60

def print_banner():
    print(DIVIDER)
    print("  CourseBot — SWE015 NLP Chatbot")
    print("  Techniques: TF-IDF · N-grams · Edit Distance")
    print("              Regex / ELIZA · Intent Detection · Stemming")
    print(DIVIDER)
    print("  Type 'help' for usage hints | 'exit' to quit")
    print(DIVIDER)

def print_debug(intent, confidence, method):
    print(f"  [DEBUG] intent={intent}  confidence={confidence:.3f}  method={method}")
    print()


# ──────────────────────────────────────────────
# 10. MAIN LOOP
# ──────────────────────────────────────────────
def main():
    print_banner()
    debug_mode = False  # toggle with 'debug on' / 'debug off'

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break

        if not user_input:
            continue

        # ── Meta-commands ──
        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            print("Bot: Goodbye! Have a great day! 👋")
            break

        if user_input.lower() == "debug on":
            debug_mode = True
            print("Bot: Debug mode enabled.")
            continue

        if user_input.lower() == "debug off":
            debug_mode = False
            print("Bot: Debug mode disabled.")
            continue

        # ── Get response ──
        answer, intent, confidence, method = get_response(user_input)

        print(f"\nBot: {answer}")

        if debug_mode:
            print_debug(intent, confidence, method)


if __name__ == "__main__":
    main()