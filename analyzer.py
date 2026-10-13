"""
analyzer.py - logika analisis konten sosial media.

Fitur:
  1. Statistik dasar (jumlah kata, panjang teks, hashtag, mention, dll.)
  2. Analisis sentimen (Positif / Negatif / Netral) dengan VADER atau TextBlob
  3. Frekuensi kata / topik yang sering muncul (unigram & bigram)

Catatan bahasa:
  VADER dan TextBlob dibuat untuk bahasa Inggris. Agar teks Indonesia ikut
  terbaca, VADER di sini diperluas dengan kamus kata Indonesia (_ID_LEXICON)
  dan kata negasi Indonesia (tidak, bukan, gak, ...). TextBlob tidak bisa
  diperluas dengan cara itu, jadi untuk teks Indonesia gunakan opsi
  translate=True (butuh paket deep-translator & internet).
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from textblob import TextBlob
from vaderSentiment import vaderSentiment as _vs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------------------------------
# Regex & stopword
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2B06\u2764]")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)|\n+")

STOPWORDS_ID = set("""
yang dan di ke dari untuk dengan ini itu atau pada juga ada akan saya aku kamu
kita kami mereka dia ia nya lah kah pun sih dong deh nih tuh lagi sudah udah
belum telah masih sangat sekali banget bisa dapat harus mau ingin karena
sebagai dalam oleh agar supaya jadi kalau kalo jika bahwa tapi tetapi namun
sama semua para setiap saat ketika sebuah suatu hanya cuma yg dg utk gak ga
nggak tidak tak bukan jangan apa siapa kapan dimana mana bagaimana gimana
kenapa mengapa lebih sudah pas biar yuk ayo kok ya
""".split())

STOPWORDS_EN = set("""
the a an and or but if then so of to in on at by for with from as is are was
were be been being it its this that these those i you he she we they them my
your our their me him her us not no do does did done have has had will would
can could should just very really more most than too also about into over
out up down what who when where why how all any some
""".split())

STOPWORDS = STOPWORDS_ID | STOPWORDS_EN

# --------------------------------------------------------------------------
# Perluasan VADER untuk bahasa Indonesia (skala -4 s.d. +4)
# --------------------------------------------------------------------------
_ID_LEXICON = {
    # positif
    "bagus": 2.5, "keren": 2.5, "kece": 2.3, "mantap": 3.0, "suka": 2.0,
    "senang": 2.5, "hebat": 3.0, "terbaik": 3.0, "cinta": 3.0, "bahagia": 3.0,
    "puas": 2.5, "memuaskan": 2.8, "murah": 1.5, "mudah": 1.8, "seru": 2.5,
    "asyik": 2.3, "indah": 2.5, "cepat": 1.5, "ramah": 2.0, "rekomen": 2.0,
    "recommended": 2.5, "worth": 2.0, "bersyukur": 2.5, "makasih": 2.0,
    "terimakasih": 2.0, "sukses": 2.8, "juara": 3.0, "nyaman": 2.0,
    "enak": 2.2, "lezat": 2.8, "gratis": 1.5, "untung": 1.8, "oke": 1.5,
    "menarik": 2.0, "bermanfaat": 2.3, "inspiratif": 2.8, "bangga": 2.5,
    "semangat": 2.2, "berkualitas": 2.5, "luarbiasa": 3.0, "cakep": 2.2,
    # negatif
    "buruk": -2.5, "jelek": -2.5, "benci": -3.0, "sedih": -2.5, "marah": -2.5,
    "rugi": -2.0, "gagal": -2.5, "susah": -1.8, "sulit": -1.5, "kecewa": -2.8,
    "mengecewakan": -3.0, "parah": -2.8, "takut": -2.0, "bohong": -2.5,
    "bohongan": -2.5, "penipuan": -3.2, "tipu": -3.0, "lambat": -1.5,
    "lelet": -2.0, "mahal": -1.5, "menyebalkan": -2.8, "kesal": -2.5,
    "capek": -1.5, "bosan": -1.8, "rusak": -2.5, "sampah": -3.0,
    "payah": -2.5, "zonk": -2.5, "nyesel": -2.5, "menyesal": -2.5,
    "toxic": -2.8, "ribet": -1.8, "bahaya": -2.3, "jahat": -3.0,
}
_ID_NEGATIONS = ["tidak", "tak", "bukan", "jangan", "belum", "gak", "ga",
                 "nggak", "enggak", "ngga"]


def _build_vader() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(_ID_LEXICON)
    for neg in _ID_NEGATIONS:          # daftar negasi VADER bersifat global
        if neg not in _vs.NEGATE:
            _vs.NEGATE.append(neg)
    return sia


_VADER = _build_vader()


# --------------------------------------------------------------------------
# 1. Statistik dasar
# --------------------------------------------------------------------------
def basic_stats(text: str) -> dict:
    """Statistik dasar satu teks."""
    text = text or ""
    words = WORD_RE.findall(text)
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    n_words = len(words)
    return {
        "karakter": len(text),
        "karakter_tanpa_spasi": len(re.sub(r"\s", "", text)),
        "kata": n_words,
        "kalimat": len(sentences),
        "rata2_panjang_kata": round(sum(map(len, words)) / n_words, 2) if n_words else 0,
        "rata2_kata_per_kalimat": round(n_words / len(sentences), 2) if sentences else 0,
        "hashtag": len(HASHTAG_RE.findall(text)),
        "mention": len(MENTION_RE.findall(text)),
        "url": len(URL_RE.findall(text)),
        "emoji": len(EMOJI_RE.findall(text)),
    }


# --------------------------------------------------------------------------
# 2. Analisis sentimen
# --------------------------------------------------------------------------
def translate_to_english(text: str) -> str:
    """Terjemahkan ke Inggris (opsional). Jika gagal, kembalikan teks asli."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text[:4500])
    except Exception:
        return text


def _label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "Positif"
    if score <= -threshold:
        return "Negatif"
    return "Netral"


_CLAUSE_RE = re.compile(r"[,;.!?\n]+")


def sentiment_vader(text: str) -> dict:
    """
    Skor VADER per klausa (dipisah , ; . ! ?) lalu dirata-rata.
    Ini mencegah negasi ("tidak bagus, saya kecewa") 'bocor' ke klausa berikutnya.
    """
    text = text or ""
    full = _VADER.polarity_scores(text)
    clauses = [c.strip() for c in _CLAUSE_RE.split(text) if c.strip()]
    scores = [_VADER.polarity_scores(c)["compound"] for c in clauses]
    scores = [x for x in scores if x != 0]
    compound = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {"score": compound, "label": _label(compound, 0.05),
            "pos": full["pos"], "neu": full["neu"], "neg": full["neg"]}


def sentiment_textblob(text: str, translate: bool = False) -> dict:
    src = translate_to_english(text) if translate else (text or "")
    sent = TextBlob(src).sentiment
    return {"score": round(sent.polarity, 4), "label": _label(sent.polarity, 0.1),
            "subjectivity": round(sent.subjectivity, 4)}


def sentiment(text: str, engine: str = "VADER", translate: bool = False) -> dict:
    if engine.upper() == "TEXTBLOB":
        return sentiment_textblob(text, translate)
    return sentiment_vader(text)


def analyze_texts(texts: list[str], engine: str = "VADER",
                  translate: bool = False) -> pd.DataFrame:
    """Analisis banyak postingan -> DataFrame (1 baris = 1 postingan)."""
    rows = []
    for t in texts:
        st = basic_stats(t)
        se = sentiment(t, engine, translate)
        rows.append({
            "teks": t, "karakter": st["karakter"], "kata": st["kata"],
            "kalimat": st["kalimat"], "hashtag": st["hashtag"],
            "mention": st["mention"], "emoji": st["emoji"],
            "skor": se["score"], "sentimen": se["label"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Frekuensi kata / topik
# --------------------------------------------------------------------------
def tokenize(text: str, extra_stopwords: set[str] | None = None,
             min_len: int = 3) -> list[str]:
    """Pecah teks jadi kata bersih (tanpa URL, mention, hashtag, stopword)."""
    stop = STOPWORDS | (extra_stopwords or set())
    text = URL_RE.sub(" ", text or "")
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    return [w for w in WORD_RE.findall(text.lower())
            if len(w) >= min_len and w not in stop]


def word_frequency(texts: list[str], top_n: int = 20, ngram: int = 1,
                   extra_stopwords: set[str] | None = None) -> pd.DataFrame:
    """Kata (ngram=1) atau pasangan kata (ngram=2) yang paling sering muncul."""
    counter: Counter = Counter()
    for t in texts:
        toks = tokenize(t, extra_stopwords)
        if ngram == 1:
            counter.update(toks)
        else:
            counter.update(" ".join(toks[i:i + ngram])
                           for i in range(len(toks) - ngram + 1))
    return pd.DataFrame(counter.most_common(top_n), columns=["kata", "frekuensi"])


def top_hashtags(texts: list[str], top_n: int = 15) -> pd.DataFrame:
    counter = Counter(h.lower() for t in texts for h in HASHTAG_RE.findall(t or ""))
    return pd.DataFrame(counter.most_common(top_n), columns=["hashtag", "frekuensi"])


# --------------------------------------------------------------------------
# Uji cepat dari terminal:  python analyzer.py "Produk ini bagus banget! #review"
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    demo = " ".join(sys.argv[1:]) or "Produk ini bagus banget, saya suka! Tapi pengirimannya lambat. #review"
    print(json.dumps(basic_stats(demo), indent=2, ensure_ascii=False))
    print("VADER   :", sentiment_vader(demo))
    print("TextBlob:", sentiment_textblob(demo))
    print(word_frequency([demo], top_n=10))
