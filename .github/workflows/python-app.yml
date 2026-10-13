"""
app.py - Dashboard analisis konten sosial media (Streamlit).
Jalankan:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import (analyze_texts, basic_stats, top_hashtags, word_frequency)

st.set_page_config(page_title="Analisis Konten Sosmed", page_icon="📊", layout="wide")

SAMPLE = """Produk ini bagus banget, saya suka! Pengiriman cepat dan penjualnya ramah 😍 #review #belanja
Kecewa parah, barang rusak dan penjual tidak mau tanggung jawab. Jangan beli di sini! #penipuan
Hari ini ikut seminar tentang marketing digital, materinya bermanfaat dan inspiratif #marketing #belajar
Love this new update, the app is so fast and easy to use! #tech #update
Harga mahal tapi kualitasnya biasa saja, kurang worth it menurut saya #review
Konten kreator harus rajin bikin video pendek, algoritma sosmed suka konten yang konsisten #konten #tips
Pelayanannya lambat banget, nunggu satu jam dan hasilnya mengecewakan #kecewa
Terima kasih semuanya, event kemarin sukses dan seru banget! 🎉 #event #komunitas
The delivery was terrible and the support team never replied #complaint
Tips marketing digital: konsisten posting, kenali audiens, dan analisa konten secara rutin #marketing #tips"""

COLORS = {"Positif": "#16a34a", "Netral": "#9ca3af", "Negatif": "#dc2626"}
ORDER = ["Positif", "Netral", "Negatif"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    engine = st.radio("Mesin sentimen", ["VADER", "TextBlob"],
                      help="VADER sudah diperluas dengan kamus Indonesia. "
                           "TextBlob hanya paham Inggris.")
    translate = st.checkbox("Terjemahkan ke Inggris dulu", value=False,
                            disabled=engine != "TextBlob",
                            help="Untuk TextBlob + teks Indonesia. Butuh deep-translator & internet.")
    unit = st.radio("Unit kata kunci", ["Kata", "Bigram (2 kata)"])
    top_n = st.slider("Jumlah kata teratas", 5, 30, 15)
    extra = st.text_input("Stopword tambahan (pisahkan koma)", "")
    extra_stop = {w.strip().lower() for w in extra.split(",") if w.strip()}

# ---------------------------------------------------------------- input
st.title("📊 Analisis Konten Sosial Media")
st.caption("Statistik dasar • Analisis sentimen • Kata kunci • Visualisasi")

src = st.radio("Sumber data", ["Ketik / tempel teks", "Upload CSV"], horizontal=True)
texts: list[str] = []

if src == "Ketik / tempel teks":
    use_sample = st.checkbox("Pakai contoh data", value=True)
    raw = st.text_area("Satu baris = satu postingan", value=SAMPLE if use_sample else "",
                       height=220)
    texts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
else:
    up = st.file_uploader("File CSV", type="csv")
    if up:
        df_in = pd.read_csv(up)
        col = st.selectbox("Kolom berisi teks", df_in.columns)
        texts = df_in[col].dropna().astype(str).tolist()

if not texts:
    st.info("Masukkan teks atau upload CSV untuk mulai.")
    st.stop()


@st.cache_data(show_spinner="Menganalisis...")
def run(texts: tuple, engine: str, translate: bool) -> pd.DataFrame:
    return analyze_texts(list(texts), engine, translate)


df = run(tuple(texts), engine, translate)

# ---------------------------------------------------------------- ringkasan
c = st.columns(5)
c[0].metric("Jumlah postingan", len(df))
c[1].metric("Total kata", int(df["kata"].sum()))
c[2].metric("Rata-rata kata/post", round(df["kata"].mean(), 1))
c[3].metric("Rata-rata karakter", round(df["karakter"].mean(), 1))
c[4].metric("Rata-rata skor sentimen", round(df["skor"].mean(), 3))

tab1, tab2, tab3, tab4 = st.tabs(["📏 Statistik", "😊 Sentimen", "🔑 Kata kunci", "🗂️ Data"])

# ---------------------------------------------------------------- statistik
with tab1:
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x="kata", nbins=15, title="Distribusi jumlah kata per postingan",
                                labels={"kata": "Jumlah kata"}), width="stretch")
    r.plotly_chart(px.histogram(df, x="karakter", nbins=15, title="Distribusi panjang teks (karakter)",
                                labels={"karakter": "Karakter"}), width="stretch")
    totals = pd.DataFrame({
        "elemen": ["Hashtag", "Mention", "Emoji"],
        "jumlah": [df["hashtag"].sum(), df["mention"].sum(), df["emoji"].sum()]})
    st.plotly_chart(px.bar(totals, x="elemen", y="jumlah", title="Total hashtag, mention & emoji",
                           text="jumlah"), width="stretch")

# ---------------------------------------------------------------- sentimen
with tab2:
    dist = (df["sentimen"].value_counts().reindex(ORDER, fill_value=0)
            .rename_axis("sentimen").reset_index(name="jumlah"))
    l, r = st.columns(2)
    l.plotly_chart(px.pie(dist, names="sentimen", values="jumlah", hole=0.45,
                          color="sentimen", color_discrete_map=COLORS,
                          title="Proporsi sentimen"), width="stretch")
    r.plotly_chart(px.histogram(df, x="skor", nbins=20, color="sentimen",
                                color_discrete_map=COLORS, category_orders={"sentimen": ORDER},
                                title="Sebaran skor sentimen (-1 s.d. +1)"),
                   width="stretch")
    dfp = df.reset_index().assign(post=lambda d: "Post " + (d["index"] + 1).astype(str))
    st.plotly_chart(px.bar(dfp, x="post", y="skor", color="sentimen",
                           color_discrete_map=COLORS, category_orders={"sentimen": ORDER},
                           hover_data=["teks"], title="Skor sentimen tiap postingan"),
                    width="stretch")

# ---------------------------------------------------------------- kata kunci
with tab3:
    ng = 1 if unit == "Kata" else 2
    wf = word_frequency(texts, top_n, ng, extra_stop)
    hf = top_hashtags(texts, top_n)
    l, r = st.columns(2)
    if wf.empty:
        l.warning("Belum ada kata yang cukup untuk dihitung.")
    else:
        l.plotly_chart(px.bar(wf.iloc[::-1], x="frekuensi", y="kata", orientation="h",
                              title=f"{unit} paling sering muncul"), width="stretch")
    if hf.empty:
        r.info("Tidak ada hashtag.")
    else:
        r.plotly_chart(px.bar(hf.iloc[::-1], x="frekuensi", y="hashtag", orientation="h",
                              title="Hashtag teratas"), width="stretch")
    try:
        from wordcloud import WordCloud
        freq = dict(word_frequency(texts, 100, 1, extra_stop).values)
        if freq:
            st.subheader("Word cloud")
            st.image(WordCloud(width=1000, height=400, background_color="white")
                     .generate_from_frequencies(freq).to_array())
    except ImportError:
        st.caption("Pasang `wordcloud` untuk menampilkan word cloud.")

# ---------------------------------------------------------------- data
with tab4:
    st.dataframe(df, width="stretch")
    st.download_button("⬇️ Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "hasil_analisis.csv", "text/csv")
    with st.expander("Statistik detail satu teks"):
        pick = st.selectbox("Pilih postingan", range(len(texts)),
                            format_func=lambda i: f"{i + 1}. {texts[i][:60]}")
        st.json(basic_stats(texts[pick]))
