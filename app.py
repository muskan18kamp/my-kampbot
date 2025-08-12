import json
import re
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS_PATH = Path("intents.json")

app = Flask(__name__, static_folder="static")
CORS(app)  # allow frontend requests

# ------- Model state -------
intents = []               # list of intent dicts (as loaded from intents.json)
pattern_to_intent = []     # index -> intent_name
corpus = []                # list of pattern strings
vectorizer = None
X = None                   # TF-IDF matrix over corpus

def _compile_regex(pattern: str):
    """Compile a regex pattern safely; return None if invalid."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None

def load_intents():
    """
    Load intents from intents.json.
    Also compiles any 'regex' present in each intent into '_regex' field for fast matching.
    Rebuilds fuzzy & TF-IDF structures from 'patterns'.
    """
    global intents, pattern_to_intent, corpus, vectorizer, X

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        # Support both formats:
        # 1) A plain list: [ {intent, patterns, regex, response}, ... ]
        # 2) An object with key "intents": {"intents": [ ... ]}
        raw = json.load(f)
        if isinstance(raw, dict) and "intents" in raw:
            intents = raw["intents"]
        else:
            intents = raw

    # Pre-compile regex from JSON (if provided)
    for it in intents:
        rgx = it.get("regex")
        if isinstance(rgx, str):
            it["_regex"] = _compile_regex(rgx)
        elif isinstance(rgx, list):
            # If regex is a list, compile each and store
            it["_regex_list"] = [r for r in ( _compile_regex(p) for p in rgx ) if r is not None]
        else:
            it["_regex"] = None

    # Build corpus for fuzzy + TF-IDF from patterns
    pattern_to_intent = []
    corpus = []
    for it in intents:
        for p in it.get("patterns", []):
            if not isinstance(p, str):
                continue
            corpus.append(p.lower())
            pattern_to_intent.append(it.get("intent"))

    # Train TF-IDF on patterns corpus
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if corpus:
        X = vectorizer.fit_transform(corpus)
    else:
        X = None

# initial load
load_intents()

def find_intent(user_text, fuzzy_threshold=70, tfidf_threshold=0.35):
    text = (user_text or "").lower().strip()
    if not text:
        return None

    # 1) Exact keyword/substring match against patterns
    for it in intents:
        for p in it.get("patterns", []):
            if isinstance(p, str) and p.lower() in text:
                return it

    # 2) Regex match — pulled directly from intents.json
    for it in intents:
        # single regex
        cre = it.get("_regex")
        if cre and cre.search(text):
            return it
        # multiple regex
        for cre2 in it.get("_regex_list", []) or []:
            if cre2.search(text):
                return it

    # 3) Fuzzy match over patterns
    all_patterns = [p for it in intents for p in it.get("patterns", []) if isinstance(p, str)]
    if all_patterns:
        match = process.extractOne(text, all_patterns, scorer=fuzz.token_set_ratio)
        if match:
            _, score, idx = match
            if score >= fuzzy_threshold:
                intent_name = pattern_to_intent[idx]
                # find the corresponding intent dict
                for it in intents:
                    if it.get("intent") == intent_name:
                        return it

    # 4) TF-IDF semantic similarity over patterns
    if X is not None and vectorizer is not None:
        q = vectorizer.transform([text])
        sims = cosine_similarity(q, X)[0]
        best_idx = sims.argmax()
        best_score = float(sims[best_idx])
        if best_score >= tfidf_threshold:
            intent_name = pattern_to_intent[best_idx]
            for it in intents:
                if it.get("intent") == intent_name:
                    return it

    return None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = data.get("message", "")
    it = find_intent(message)
    if it:
        return jsonify({"response": it.get("response", "")})
    return jsonify({"response": "Sorry, I didn't get it. Please type an appropriate message."})

@app.route("/reload", methods=["POST"])
def reload_route():
    load_intents()
    return jsonify({"ok": True})

# Serve the HTML UI
@app.route("/")
def serve_ui():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

