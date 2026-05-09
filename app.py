import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ARM System — Apriori",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; }

    .metric-card {
        background: #0f3460;
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0; font-size: 0.85rem; }

    .rule-card {
        background: #16213e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #0f3460;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #1a4a7a;
        color: #a8b2d8;
        font-size: 0.9rem;
    }
    .step-box {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
    }
    .step-box h4 { color: #e94560; margin: 0 0 0.3rem; font-size: 0.95rem; }
    .step-box p  { color: #a8b2d8; margin: 0; font-size: 0.85rem; }

    .rak-a {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
    }
    .rak-b {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
    }
    .recom-card {
        background: #16213e;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .pred-card {
        background: #16213e;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }

    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    div[data-testid="metric-container"] {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  APRIORI ALGORITHM
# ─────────────────────────────────────────
def get_support(itemset: frozenset, transactions: list) -> float:
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def get_support_count(itemset: frozenset, transactions: list) -> int:
    return sum(1 for t in transactions if itemset.issubset(t))

def generate_candidates(prev_frequent: list, k: int) -> list:
    candidates = []
    prev_list = [sorted(list(fs)) for fs in prev_frequent]
    for i in range(len(prev_list)):
        for j in range(i+1, len(prev_list)):
            merged = sorted(set(prev_list[i]) | set(prev_list[j]))
            if len(merged) == k:
                cand = frozenset(merged)
                if cand not in candidates:
                    candidates.append(cand)
    return candidates

def apriori(transactions: list, min_support: float):
    all_items  = frozenset(item for t in transactions for item in t)
    candidates = [frozenset([item]) for item in all_items]

    frequent_itemsets = {}
    support_data      = {}
    k = 1

    while candidates:
        freq = {}
        for cand in candidates:
            sup = get_support(cand, transactions)
            support_data[cand] = sup
            if sup >= min_support:
                freq[cand] = sup
        if not freq:
            break
        frequent_itemsets.update(freq)
        k += 1
        candidates = generate_candidates(list(freq.keys()), k)
        if k > 6:
            break

    return frequent_itemsets, support_data

def generate_rules(frequent_itemsets: dict, min_confidence: float, min_lift: float):
    rules = []
    for itemset, sup in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent  = frozenset(antecedent)
                consequent  = itemset - antecedent
                ant_sup     = frequent_itemsets.get(antecedent, 0)
                con_sup     = frequent_itemsets.get(consequent, 0)
                if ant_sup == 0 or con_sup == 0:
                    continue
                confidence  = sup / ant_sup
                lift        = confidence / con_sup
                leverage    = sup - (ant_sup * con_sup)
                conviction  = (1 - con_sup) / (1 - confidence) if confidence < 1 else float('inf')
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        'antecedent':  ', '.join(sorted(antecedent)),
                        'consequent':  ', '.join(sorted(consequent)),
                        'support':     round(sup, 4),
                        'confidence':  round(confidence, 4),
                        'lift':        round(lift, 4),
                        'leverage':    round(leverage, 4),
                        'conviction':  round(conviction, 4) if conviction != float('inf') else 9999,
                        'ant_support': round(ant_sup, 4),
                        'con_support': round(con_sup, 4),
                    })
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values('lift', ascending=False).reset_index(drop=True)
    return rules_df


# ─────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────
def load_data(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    return None

def prepare_transactions(df: pd.DataFrame, invoice_col: str, item_col: str):
    df_clean = df.dropna(subset=[item_col]).copy()
    df_clean[item_col] = df_clean[item_col].astype(str)
    grouped  = df_clean.groupby(invoice_col)[item_col].apply(lambda x: frozenset(x.unique()))
    return grouped.tolist()


# ─────────────────────────────────────────
#  SISTEM 1: REKOMENDASI PRODUK
# ─────────────────────────────────────────
def get_recommendations(selected_items: list, rules_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Cari rekomendasi produk berdasarkan item yang dipilih pelanggan."""
    if rules_df.empty or not selected_items:
        return pd.DataFrame()

    selected_set = set(selected_items)
    recommendations = []

    for _, row in rules_df.iterrows():
        antecedents = set(a.strip() for a in row['antecedent'].split(','))
        consequents = set(c.strip() for c in row['consequent'].split(','))

        # Cek apakah antecedent ada dalam item yang dipilih
        overlap = antecedents & selected_set
        if overlap:
            for item in consequents:
                if item not in selected_set:
                    recommendations.append({
                        'Produk Rekomendasi': item,
                        'Karena Beli': ', '.join(overlap),
                        'Confidence (%)': round(row['confidence'] * 100, 1),
                        'Lift': round(row['lift'], 3),
                        'Support (%)': round(row['support'] * 100, 1),
                    })

    if not recommendations:
        return pd.DataFrame()

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values(['Lift', 'Confidence (%)'], ascending=False)
    rec_df = rec_df.drop_duplicates(subset=['Produk Rekomendasi']).head(top_n)
    return rec_df.reset_index(drop=True)


# ─────────────────────────────────────────
#  SISTEM 2: SPK TATA LETAK RAK (Rak A / Rak B)
# ─────────────────────────────────────────
def classify_shelf(frequent_itemsets: dict, rules_df: pd.DataFrame,
                   support_threshold: float = 0.15) -> dict:
    """
    Klasifikasi item ke Rak A (sering dibeli/strategis) atau Rak B (jarang).
    Skor = 0.5 * support_normalized + 0.3 * avg_confidence + 0.2 * avg_lift_norm
    """
    if not frequent_itemsets:
        return {}

    l1 = {list(k)[0]: v for k, v in frequent_itemsets.items() if len(k) == 1}
    if not l1:
        return {}

    max_sup = max(l1.values())
    min_sup = min(l1.values())
    sup_range = max_sup - min_sup if max_sup != min_sup else 1

    item_stats = {}
    for item, sup in l1.items():
        item_stats[item] = {
            'support': sup,
            'support_norm': (sup - min_sup) / sup_range,
            'avg_confidence': 0.0,
            'avg_lift': 0.0,
            'rule_count': 0,
        }

    if not rules_df.empty:
        for _, row in rules_df.iterrows():
            all_items = [a.strip() for a in row['antecedent'].split(',')] + \
                        [c.strip() for c in row['consequent'].split(',')]
            for item in all_items:
                if item in item_stats:
                    n = item_stats[item]['rule_count']
                    item_stats[item]['avg_confidence'] = (
                        item_stats[item]['avg_confidence'] * n + row['confidence']) / (n + 1)
                    item_stats[item]['avg_lift'] = (
                        item_stats[item]['avg_lift'] * n + row['lift']) / (n + 1)
                    item_stats[item]['rule_count'] += 1

    # Normalisasi lift
    lifts = [v['avg_lift'] for v in item_stats.values() if v['avg_lift'] > 0]
    max_lift = max(lifts) if lifts else 1
    min_lift = min(lifts) if lifts else 0
    lift_range = max_lift - min_lift if max_lift != min_lift else 1

    for item in item_stats:
        raw_lift = item_stats[item]['avg_lift']
        item_stats[item]['lift_norm'] = (raw_lift - min_lift) / lift_range if raw_lift > 0 else 0

    # Hitung skor akhir
    for item in item_stats:
        s = item_stats[item]
        s['score'] = (0.5 * s['support_norm'] +
                      0.3 * s['avg_confidence'] +
                      0.2 * s['lift_norm'])

    scores = sorted(item_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    median_score = np.median([v['score'] for _, v in scores])

    result = {}
    for item, stats in scores:
        rak = 'A' if stats['score'] >= median_score else 'B'
        result[item] = {**stats, 'rak': rak}

    return result


# ─────────────────────────────────────────
#  SISTEM 3: PREDIKSI PERILAKU PEMBELIAN
# ─────────────────────────────────────────
def predict_next_purchase(cart_items: list, rules_df: pd.DataFrame,
                          frequent_itemsets: dict, top_n: int = 5) -> pd.DataFrame:
    """
    Prediksi item berikutnya yang kemungkinan besar akan dibeli.
    Menggunakan confidence tertinggi sebagai probabilitas prediksi.
    """
    if rules_df.empty or not cart_items:
        return pd.DataFrame()

    cart_set = set(cart_items)
    predictions = defaultdict(lambda: {'prob': 0.0, 'triggers': [], 'lift': 0.0, 'support': 0.0})

    for _, row in rules_df.iterrows():
        antecedents = set(a.strip() for a in row['antecedent'].split(','))
        consequents = set(c.strip() for c in row['consequent'].split(','))

        if antecedents.issubset(cart_set):
            for item in consequents:
                if item not in cart_set:
                    if row['confidence'] > predictions[item]['prob']:
                        predictions[item]['prob'] = row['confidence']
                        predictions[item]['lift'] = row['lift']
                        predictions[item]['support'] = row['support']
                    if ', '.join(sorted(antecedents)) not in predictions[item]['triggers']:
                        predictions[item]['triggers'].append(', '.join(sorted(antecedents)))

    if not predictions:
        return pd.DataFrame()

    rows = []
    for item, data in predictions.items():
        prob = data['prob']
        kategori = (
            '🔴 Sangat Tinggi' if prob >= 0.8 else
            '🟠 Tinggi'        if prob >= 0.6 else
            '🟡 Sedang'        if prob >= 0.4 else
            '🟢 Rendah'
        )
        rows.append({
            'Item Prediksi': item,
            'Probabilitas Beli (%)': round(prob * 100, 1),
            'Kategori': kategori,
            'Lift': round(data['lift'], 3),
            'Support (%)': round(data['support'] * 100, 1),
            'Dipicu Oleh': '; '.join(data['triggers'][:2]),
        })

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.sort_values('Probabilitas Beli (%)', ascending=False).head(top_n)
    return pred_df.reset_index(drop=True)


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    st.markdown("---")

    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload file transaksi (CSV / Excel)",
        type=['csv', 'xlsx', 'xls']
    )
    st.caption("Format: setiap baris = satu item dalam transaksi. Harus ada kolom invoice & produk.")

    st.markdown("---")

    st.markdown("### 🔧 Kolom Data")
    invoice_col = st.text_input("Kolom Invoice/Transaksi", value="Invoice")
    item_col    = st.text_input("Kolom Item/Produk",        value="Description")

    st.markdown("---")

    st.markdown("### 🎯 Parameter Apriori")
    min_support    = st.slider("Minimum Support",    0.01, 0.80, 0.10, 0.01,
                               help="Frekuensi minimum suatu itemset muncul dalam transaksi")
    min_confidence = st.slider("Minimum Confidence", 0.10, 1.00, 0.50, 0.05,
                               help="Probabilitas minimum konsekuensi muncul bila anteseden ada")
    min_lift       = st.slider("Minimum Lift",       1.00, 5.00, 1.00, 0.10,
                               help="Lift > 1 berarti ada korelasi positif antar item")

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Mining", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ Tentang Lift")
    st.info("**Lift > 1** → Item saling melengkapi\n\n**Lift = 1** → Item tidak berkaitan\n\n**Lift < 1** → Item saling menggantikan")


# ─────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔗 Association Rule Mining System</h1>
    <p>Implementasi Algoritma Apriori · Rekomendasi · SPK Rak · Prediksi Perilaku</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Data Eksplorasi",
    "⚙️ Proses Mining",
    "📦 Frequent Itemsets",
    "📜 Association Rules",
    "📈 Visualisasi",
    "🛒 Sistem Rekomendasi",
    "🗂️ SPK Tata Letak Rak",
    "🔮 Prediksi Perilaku",
])


# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
df = load_data(uploaded_file)

if df is None:
    st.warning("⬅️ Silakan upload file transaksi (CSV/Excel) melalui sidebar untuk memulai.")
    st.stop()

# Validate columns
if invoice_col not in df.columns or item_col not in df.columns:
    available = ', '.join(df.columns.tolist())
    st.error(f"Kolom `{invoice_col}` atau `{item_col}` tidak ditemukan.\n\nKolom tersedia: **{available}**")
    st.stop()

transactions = prepare_transactions(df, invoice_col, item_col)


# ─────────────────────────────────────────
#  TAB 1 — DATA EKSPLORASI
# ─────────────────────────────────────────
with tab1:
    st.subheader("📊 Eksplorasi Dataset")

    n_txn    = df[invoice_col].nunique()
    n_items  = df[item_col].nunique()
    avg_size = df.groupby(invoice_col)[item_col].nunique().mean()
    total_r  = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧾 Total Transaksi",   f"{n_txn:,}")
    c2.metric("🏷️  Produk Unik",       f"{n_items:,}")
    c3.metric("📦 Rata-rata Item/Txn", f"{avg_size:.1f}")
    c4.metric("📝 Total Baris Data",   f"{total_r:,}")

    st.markdown("---")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 👀 Sampel Data Mentah")
        st.dataframe(df.head(20), use_container_width=True, height=280)

    with col_b:
        st.markdown("#### 📦 Distribusi Jumlah Item per Transaksi")
        txn_sizes = df.groupby(invoice_col)[item_col].nunique()
        fig_dist  = px.histogram(
            txn_sizes, nbins=15,
            labels={'value': 'Jumlah Item', 'count': 'Frekuensi'},
            color_discrete_sequence=['#e94560']
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a8b2d8', showlegend=False, height=280,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_dist.update_xaxes(gridcolor='#1a4a7a')
        fig_dist.update_yaxes(gridcolor='#1a4a7a')
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 🏆 Top 15 Produk Terlaris")
    top_items = df[item_col].value_counts().head(15).reset_index()
    top_items.columns = ['Produk', 'Jumlah Transaksi']
    fig_top = px.bar(
        top_items, x='Jumlah Transaksi', y='Produk', orientation='h',
        color='Jumlah Transaksi', color_continuous_scale='Reds'
    )
    fig_top.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8b2d8', showlegend=False, height=380,
        margin=dict(l=10, r=10, t=10, b=10), yaxis={'categoryorder': 'total ascending'}
    )
    fig_top.update_xaxes(gridcolor='#1a4a7a')
    fig_top.update_yaxes(gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_top, use_container_width=True)

    if 'Country' in df.columns:
        st.markdown("#### 🌍 Distribusi Transaksi per Negara")
        country_counts = df.groupby('Country')[invoice_col].nunique().reset_index()
        country_counts.columns = ['Negara', 'Jumlah Transaksi']
        fig_country = px.pie(
            country_counts, names='Negara', values='Jumlah Transaksi',
            color_discrete_sequence=['#e94560', '#0f3460', '#533483', '#2ecc71', '#16213e']
        )
        fig_country.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#a8b2d8', height=320
        )
        st.plotly_chart(fig_country, use_container_width=True)


# ─────────────────────────────────────────
#  TAB 2 — PROSES MINING
# ─────────────────────────────────────────
with tab2:
    st.subheader("⚙️ Proses Algoritma Apriori")

    st.markdown("""
    <div class="info-box">
    <b>Algoritma Apriori</b> adalah metode klasik dalam Association Rule Mining yang bekerja berdasarkan
    prinsip <i>anti-monotone</i>: jika suatu itemset tidak frequent, maka semua supersetnya pun tidak frequent.
    Algoritma ini bekerja secara iteratif dari 1-itemset hingga tidak ada lagi frequent itemset yang bisa dibentuk.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Langkah-Langkah Algoritma")

    steps = [
        ("1. Preprocessing Data", "Konversi data transaksi ke format list of sets. Setiap transaksi direpresentasikan sebagai himpunan item."),
        ("2. Generate Kandidat C1", "Buat kandidat 1-itemset dari semua item unik yang ada dalam seluruh transaksi."),
        ("3. Hitung Support C1 → L1", f"Hitung support setiap 1-itemset. Filter item dengan support ≥ {min_support:.0%}. Hasil = L1 (frequent 1-itemsets)."),
        ("4. Generate & Prune Lk", "Join Lk-1 dengan dirinya sendiri untuk menghasilkan Ck. Prune kandidat yang subset-nya tidak frequent. Hitung support, filter → Lk."),
        ("5. Generate Association Rules", f"Untuk setiap frequent itemset, hasilkan semua kemungkinan rules. Filter berdasarkan confidence ≥ {min_confidence:.0%} dan lift ≥ {min_lift:.1f}."),
    ]

    for title, desc in steps:
        st.markdown(f"""
        <div class="step-box">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 Parameter yang Digunakan")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Minimum Support",    f"{min_support:.0%}")
    pc2.metric("Minimum Confidence", f"{min_confidence:.0%}")
    pc3.metric("Minimum Lift",       f"{min_lift:.1f}")

    if run_btn:
        st.markdown("---")
        st.markdown("#### 🔄 Log Proses Mining")
        log_placeholder = st.empty()
        prog = st.progress(0)
        logs = []

        def update_log(msg):
            logs.append(f"✅ {msg}")
            log_placeholder.markdown("\n\n".join(logs))

        with st.spinner("Mining sedang berjalan..."):
            update_log(f"Memuat {len(transactions)} transaksi...")
            prog.progress(10)
            time.sleep(0.3)

            update_log("Preprocessing: konversi ke format frozenset selesai.")
            prog.progress(25)
            time.sleep(0.3)

            t0 = time.time()
            frequent_itemsets, support_data = apriori(transactions, min_support)
            elapsed_fi = time.time() - t0

            l1 = {k: v for k, v in frequent_itemsets.items() if len(k) == 1}
            l2 = {k: v for k, v in frequent_itemsets.items() if len(k) == 2}
            l3 = {k: v for k, v in frequent_itemsets.items() if len(k) == 3}

            update_log(f"L1 (1-itemsets frequent): {len(l1)} item lolos threshold.")
            prog.progress(50)
            time.sleep(0.2)

            update_log(f"L2 (2-itemsets frequent): {len(l2)} itemset ditemukan.")
            prog.progress(65)
            time.sleep(0.2)

            update_log(f"L3 (3-itemsets frequent): {len(l3)} itemset ditemukan.")
            prog.progress(75)
            time.sleep(0.2)

            update_log(f"Total frequent itemsets: {len(frequent_itemsets)} (waktu: {elapsed_fi:.2f}s)")
            prog.progress(85)

            t1 = time.time()
            rules_df = generate_rules(frequent_itemsets, min_confidence, min_lift)
            elapsed_rules = time.time() - t1

            update_log(f"Association rules ditemukan: {len(rules_df)} rules (waktu: {elapsed_rules:.3f}s)")
            prog.progress(100)

        st.success(f"🎉 Mining selesai! {len(frequent_itemsets)} frequent itemsets dan {len(rules_df)} rules berhasil ditemukan.")
        st.session_state['frequent_itemsets'] = frequent_itemsets
        st.session_state['rules_df']          = rules_df
        st.session_state['transactions']      = transactions
        st.session_state['ran']               = True

    elif 'ran' not in st.session_state:
        st.info("⬅️ Atur parameter di sidebar, lalu tekan **Jalankan Mining**.")


# ─────────────────────────────────────────
#  TAB 3 — FREQUENT ITEMSETS
# ─────────────────────────────────────────
with tab3:
    st.subheader("📦 Frequent Itemsets")

    if 'frequent_itemsets' not in st.session_state:
        st.info("Jalankan mining terlebih dahulu di tab **Proses Mining**.")
        st.stop()

    fi = st.session_state['frequent_itemsets']

    fi_rows = []
    for itemset, sup in fi.items():
        fi_rows.append({
            'Itemset': '{' + ', '.join(sorted(itemset)) + '}',
            'Ukuran': len(itemset),
            'Support (%)': round(sup * 100, 2),
            'Support Count': int(round(sup * len(transactions))),
        })
    fi_df = pd.DataFrame(fi_rows).sort_values(['Ukuran', 'Support (%)'], ascending=[True, False])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Itemsets", len(fi_df))
    c2.metric("1-Itemsets",     len(fi_df[fi_df['Ukuran'] == 1]))
    c3.metric("2-Itemsets",     len(fi_df[fi_df['Ukuran'] == 2]))
    c4.metric("3-Itemsets",     len(fi_df[fi_df['Ukuran'] == 3]))

    size_filter = st.multiselect("Filter ukuran itemset:", [1, 2, 3, 4], default=[1, 2, 3])
    filtered_fi = fi_df[fi_df['Ukuran'].isin(size_filter)] if size_filter else fi_df

    st.dataframe(
        filtered_fi.style.background_gradient(subset=['Support (%)'], cmap='Reds'),
        use_container_width=True, height=420
    )

    csv = filtered_fi.to_csv(index=False)
    st.download_button("⬇️ Download Frequent Itemsets (CSV)", csv,
                       "frequent_itemsets.csv", "text/csv")


# ─────────────────────────────────────────
#  TAB 4 — ASSOCIATION RULES
# ─────────────────────────────────────────
with tab4:
    st.subheader("📜 Association Rules")

    if 'rules_df' not in st.session_state:
        st.info("Jalankan mining terlebih dahulu di tab **Proses Mining**.")
        st.stop()

    rules_df = st.session_state['rules_df']

    if rules_df.empty:
        st.warning("Tidak ada rules yang memenuhi threshold. Coba turunkan minimum confidence atau lift.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rules",       len(rules_df))
    c2.metric("Avg Confidence",    f"{rules_df['confidence'].mean()*100:.1f}%")
    c3.metric("Avg Lift",          f"{rules_df['lift'].mean():.2f}")
    c4.metric("Max Lift",          f"{rules_df['lift'].max():.2f}")

    st.markdown("---")

    sort_col = st.selectbox("Urutkan berdasarkan:", ['lift', 'confidence', 'support', 'leverage'], index=0)
    display_df = rules_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    display_df.index += 1

    display_df['support (%)']    = (display_df['support']    * 100).round(2)
    display_df['confidence (%)'] = (display_df['confidence'] * 100).round(2)

    show_cols = ['antecedent', 'consequent', 'support (%)', 'confidence (%)', 'lift', 'leverage', 'conviction']
    st.dataframe(
        display_df[show_cols].style
            .background_gradient(subset=['lift'],           cmap='Reds')
            .background_gradient(subset=['confidence (%)'], cmap='Blues')
            .format({'lift': '{:.3f}', 'leverage': '{:.4f}', 'conviction': '{:.3f}'}),
        use_container_width=True, height=460
    )

    csv_rules = display_df[show_cols].to_csv(index=False)
    st.download_button("⬇️ Download Association Rules (CSV)", csv_rules,
                       "association_rules.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 🏆 Top 5 Rules Terbaik (berdasarkan Lift)")
    for i, row in display_df.head(5).iterrows():
        lift_color = '#2ecc71' if row['lift'] >= 2 else '#f39c12' if row['lift'] >= 1.5 else '#e74c3c'
        st.markdown(f"""
        <div class="rule-card">
            <b style="color:#e94560">IF</b> {{{row['antecedent']}}}
            &nbsp;→&nbsp;
            <b style="color:#2ecc71">THEN</b> {{{row['consequent']}}}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Support: <b>{row['support (%)']:.1f}%</b> &nbsp;
            Confidence: <b>{row['confidence (%)']:.1f}%</b> &nbsp;
            Lift: <b style="color:{lift_color}">{row['lift']:.3f}</b>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TAB 5 — VISUALISASI
# ─────────────────────────────────────────
with tab5:
    st.subheader("📈 Visualisasi Hasil Mining")

    if 'rules_df' not in st.session_state or st.session_state['rules_df'].empty:
        st.info("Jalankan mining dan pastikan ada rules yang ditemukan.")
        st.stop()

    rules_df = st.session_state['rules_df']
    fi       = st.session_state['frequent_itemsets']

    st.markdown("#### 🔵 Sebaran Support vs Confidence (ukuran = Lift)")
    fig_sc = px.scatter(
        rules_df,
        x='support', y='confidence',
        size='lift', color='lift',
        hover_data=['antecedent', 'consequent', 'lift'],
        color_continuous_scale='Reds',
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
        size_max=30
    )
    fig_sc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8b2d8', height=400
    )
    fig_sc.update_xaxes(gridcolor='#1a4a7a', tickformat='.0%')
    fig_sc.update_yaxes(gridcolor='#1a4a7a', tickformat='.0%')
    st.plotly_chart(fig_sc, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📊 Top 10 Rules — Lift")
        top10 = rules_df.head(10).copy()
        top10['rule'] = top10['antecedent'].str[:20] + ' → ' + top10['consequent'].str[:15]
        fig_lift = px.bar(
            top10, x='lift', y='rule', orientation='h',
            color='lift', color_continuous_scale='Reds',
            labels={'lift': 'Lift', 'rule': 'Rule'}
        )
        fig_lift.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a8b2d8', height=380, showlegend=False,
            yaxis={'categoryorder': 'total ascending'}, margin=dict(l=5, r=5, t=5, b=5)
        )
        fig_lift.update_xaxes(gridcolor='#1a4a7a')
        fig_lift.update_yaxes(gridcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lift, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 Top 10 Item — Support")
        l1_items = {list(k)[0]: v for k, v in fi.items() if len(k) == 1}
        top_items_sup = sorted(l1_items.items(), key=lambda x: x[1], reverse=True)[:10]
        df_sup = pd.DataFrame(top_items_sup, columns=['Item', 'Support'])
        fig_sup = px.bar(
            df_sup, x='Support', y='Item', orientation='h',
            color='Support', color_continuous_scale='Blues',
        )
        fig_sup.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#a8b2d8', height=380, showlegend=False,
            yaxis={'categoryorder': 'total ascending'}, margin=dict(l=5, r=5, t=5, b=5)
        )
        fig_sup.update_xaxes(gridcolor='#1a4a7a', tickformat='.0%')
        fig_sup.update_yaxes(gridcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sup, use_container_width=True)

    st.markdown("#### 🔥 Heatmap Confidence antar Item (Top 12)")
    top12 = [list(k)[0] for k, v in sorted(
        {k: v for k, v in fi.items() if len(k) == 1}.items(), key=lambda x: -x[1]
    )[:12]]
    matrix = pd.DataFrame(0.0, index=top12, columns=top12)
    for _, row in rules_df.iterrows():
        ants = [a.strip() for a in row['antecedent'].split(',')]
        cons = [c.strip() for c in row['consequent'].split(',')]
        for a in ants:
            for c in cons:
                if a in top12 and c in top12:
                    matrix.loc[a, c] = max(matrix.loc[a, c], row['confidence'])

    fig_heat = px.imshow(
        matrix, color_continuous_scale='Reds',
        labels=dict(x='Consequent', y='Antecedent', color='Confidence'),
        aspect='auto'
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8b2d8', height=420
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### 📉 Distribusi Nilai Lift")
    fig_lift_hist = px.histogram(
        rules_df, x='lift', nbins=20,
        color_discrete_sequence=['#e94560'],
        labels={'lift': 'Lift Value', 'count': 'Jumlah Rules'}
    )
    fig_lift_hist.add_vline(x=1.0, line_dash='dash', line_color='#a8b2d8',
                            annotation_text='Lift = 1 (no association)')
    fig_lift_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8b2d8', height=300
    )
    fig_lift_hist.update_xaxes(gridcolor='#1a4a7a')
    fig_lift_hist.update_yaxes(gridcolor='#1a4a7a')
    st.plotly_chart(fig_lift_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 6 — 🛒 SISTEM REKOMENDASI PRODUK
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🛒 Sistem Rekomendasi Produk")

    st.markdown("""
    <div class="info-box">
    <b>Sistem Rekomendasi</b> ini menggunakan association rules yang telah ditemukan untuk menyarankan
    produk tambahan berdasarkan item yang sudah dipilih pelanggan. Semakin tinggi <b>Confidence</b> dan
    <b>Lift</b>, semakin kuat rekomendasi tersebut.
    </div>
    """, unsafe_allow_html=True)

    if 'rules_df' not in st.session_state:
        st.info("⬅️ Jalankan mining terlebih dahulu di tab **Proses Mining**.")
        st.stop()

    rules_df_r = st.session_state['rules_df']

    if rules_df_r.empty:
        st.warning("Tidak ada rules. Turunkan parameter threshold lalu jalankan ulang mining.")
        st.stop()

    # Kumpulkan semua item unik dari rules
    all_items_in_rules = set()
    for _, row in rules_df_r.iterrows():
        for it in row['antecedent'].split(','):
            all_items_in_rules.add(it.strip())
        for it in row['consequent'].split(','):
            all_items_in_rules.add(it.strip())
    all_items_sorted = sorted(all_items_in_rules)

    st.markdown("#### 🛍️ Pilih Item yang Sudah Ada di Keranjang")
    selected_items = st.multiselect(
        "Pilih satu atau lebih produk:",
        options=all_items_sorted,
        help="Sistem akan merekomendasikan produk lain yang sering dibeli bersama item yang kamu pilih."
    )

    top_n_rec = st.slider("Jumlah rekomendasi:", 3, 15, 5)

    if selected_items:
        rec_df = get_recommendations(selected_items, rules_df_r, top_n=top_n_rec)

        if rec_df.empty:
            st.warning("Tidak ada rekomendasi untuk kombinasi item ini. Coba pilih item yang berbeda.")
        else:
            st.markdown(f"#### 💡 {len(rec_df)} Rekomendasi Produk untuk Kamu")

            for _, row in rec_df.iterrows():
                conf_color = '#2ecc71' if row['Confidence (%)'] >= 70 else '#f39c12' if row['Confidence (%)'] >= 50 else '#e74c3c'
                st.markdown(f"""
                <div class="recom-card">
                    <b style="color:#2ecc71; font-size:1.05rem">🏷️ {row['Produk Rekomendasi']}</b><br>
                    <span style="color:#a8b2d8; font-size:0.85rem">
                        Karena beli: <i>{row['Karena Beli']}</i>
                    </span><br>
                    Confidence: <b style="color:{conf_color}">{row['Confidence (%)']:.1f}%</b> &nbsp;|&nbsp;
                    Lift: <b>{row['Lift']:.3f}</b> &nbsp;|&nbsp;
                    Support: <b>{row['Support (%)']:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 📊 Visualisasi Kekuatan Rekomendasi")
            fig_rec = px.bar(
                rec_df,
                x='Confidence (%)',
                y='Produk Rekomendasi',
                orientation='h',
                color='Lift',
                color_continuous_scale='Greens',
                hover_data=['Karena Beli', 'Support (%)'],
                labels={'Produk Rekomendasi': 'Produk'}
            )
            fig_rec.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#a8b2d8', height=350,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=5, r=5, t=10, b=5)
            )
            fig_rec.update_xaxes(gridcolor='#1a4a7a')
            fig_rec.update_yaxes(gridcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rec, use_container_width=True)

            csv_rec = rec_df.to_csv(index=False)
            st.download_button("⬇️ Download Rekomendasi (CSV)", csv_rec,
                               "rekomendasi_produk.csv", "text/csv")
    else:
        st.info("Pilih produk di atas untuk mendapatkan rekomendasi.")

    # Statistik ringkasan rekomendasi secara umum
    st.markdown("---")
    st.markdown("#### 📌 Statistik Kekuatan Rekomendasi Keseluruhan")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rules Tersedia", len(rules_df_r))
    col2.metric("Rata-rata Confidence", f"{rules_df_r['confidence'].mean()*100:.1f}%")
    col3.metric("Rata-rata Lift", f"{rules_df_r['lift'].mean():.2f}")


# ═══════════════════════════════════════════════════════════════
#  TAB 7 — 🗂️ SPK TATA LETAK RAK (RAK A / RAK B)
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🗂️ SPK Tata Letak Rak — Rak A & Rak B")

    st.markdown("""
    <div class="info-box">
    <b>Sistem Pendukung Keputusan (SPK) Tata Letak Rak</b> ini mengklasifikasikan produk ke dalam dua kategori rak:<br><br>
    🟢 <b>Rak A (Strategis / Sering Dibeli)</b> — Item dengan support tinggi, banyak terlibat dalam rules dengan confidence dan lift baik.
    Tempatkan di area mudah dijangkau / lorong utama.<br><br>
    🔴 <b>Rak B (Jarang Dibeli / Pelengkap)</b> — Item dengan frekuensi lebih rendah. 
    Bisa ditempatkan di area pojok atau rak belakang, namun dekat dengan item Rak A yang berkorelasi.<br><br>
    Skor dihitung dari: <code>0.5 × support + 0.3 × avg_confidence + 0.2 × avg_lift</code>
    </div>
    """, unsafe_allow_html=True)

    if 'frequent_itemsets' not in st.session_state:
        st.info("⬅️ Jalankan mining terlebih dahulu di tab **Proses Mining**.")
        st.stop()

    fi_s     = st.session_state['frequent_itemsets']
    rules_s  = st.session_state['rules_df']

    shelf_data = classify_shelf(fi_s, rules_s)

    if not shelf_data:
        st.warning("Tidak cukup data untuk klasifikasi rak. Coba turunkan min support.")
        st.stop()

    rak_a = [(item, d) for item, d in shelf_data.items() if d['rak'] == 'A']
    rak_b = [(item, d) for item, d in shelf_data.items() if d['rak'] == 'B']

    # Ringkasan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Total Rak A", len(rak_a))
    c2.metric("🔴 Total Rak B", len(rak_b))
    c3.metric("Total Item",     len(shelf_data))
    c4.metric("Threshold Skor", f"{np.median([d['score'] for d in shelf_data.values()]):.3f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🟢 RAK A — Produk Strategis")
        st.caption("Tempatkan di lokasi utama / mudah dijangkau pelanggan")
        for item, data in sorted(rak_a, key=lambda x: x[1]['score'], reverse=True):
            st.markdown(f"""
            <div class="rak-a">
                <b style="color:#2ecc71">{item}</b><br>
                <span style="color:#a8b2d8; font-size:0.82rem">
                    Support: <b>{data['support']*100:.1f}%</b> &nbsp;|&nbsp;
                    Avg Conf: <b>{data['avg_confidence']*100:.1f}%</b> &nbsp;|&nbsp;
                    Avg Lift: <b>{data['avg_lift']:.2f}</b> &nbsp;|&nbsp;
                    Skor: <b>{data['score']:.3f}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### 🔴 RAK B — Produk Pendukung")
        st.caption("Tempatkan di area pojok atau rak belakang")
        for item, data in sorted(rak_b, key=lambda x: x[1]['score'], reverse=True):
            st.markdown(f"""
            <div class="rak-b">
                <b style="color:#e74c3c">{item}</b><br>
                <span style="color:#a8b2d8; font-size:0.82rem">
                    Support: <b>{data['support']*100:.1f}%</b> &nbsp;|&nbsp;
                    Avg Conf: <b>{data['avg_confidence']*100:.1f}%</b> &nbsp;|&nbsp;
                    Avg Lift: <b>{data['avg_lift']:.2f}</b> &nbsp;|&nbsp;
                    Skor: <b>{data['score']:.3f}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Visualisasi skor
    st.markdown("---")
    st.markdown("#### 📊 Perbandingan Skor Item (Rak A vs Rak B)")

    shelf_df = pd.DataFrame([
        {'Item': item, 'Rak': data['rak'], 'Skor': round(data['score'], 4),
         'Support (%)': round(data['support']*100, 2),
         'Avg Confidence (%)': round(data['avg_confidence']*100, 2),
         'Avg Lift': round(data['avg_lift'], 3)}
        for item, data in shelf_data.items()
    ]).sort_values('Skor', ascending=False)

    fig_shelf = px.bar(
        shelf_df,
        x='Skor', y='Item', orientation='h',
        color='Rak',
        color_discrete_map={'A': '#2ecc71', 'B': '#e74c3c'},
        hover_data=['Support (%)', 'Avg Confidence (%)', 'Avg Lift'],
        labels={'Item': 'Produk', 'Skor': 'Skor SPK'}
    )
    fig_shelf.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8b2d8', height=max(400, len(shelf_data) * 28),
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=5, r=5, t=10, b=5)
    )
    fig_shelf.update_xaxes(gridcolor='#1a4a7a')
    fig_shelf.update_yaxes(gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_shelf, use_container_width=True)

    # Rekomendasi penempatan berdampingan
    st.markdown("---")
    st.markdown("#### 🔗 Rekomendasi Penempatan Berdampingan")
    st.caption("Item Rak B yang berkorelasi dengan Rak A sebaiknya ditempatkan di dekat Rak A tersebut.")

    if not rules_s.empty:
        pair_recs = []
        rak_a_items = {item for item, _ in rak_a}
        rak_b_items = {item for item, _ in rak_b}

        for _, row in rules_s.iterrows():
            ants = set(a.strip() for a in row['antecedent'].split(','))
            cons = set(c.strip() for c in row['consequent'].split(','))
            a_items = ants & rak_a_items
            b_items = (ants | cons) & rak_b_items
            if a_items and b_items:
                pair_recs.append({
                    'Rak A (Anchor)': ', '.join(sorted(a_items)),
                    'Rak B (Dekatkan)': ', '.join(sorted(b_items)),
                    'Confidence (%)': round(row['confidence']*100, 1),
                    'Lift': round(row['lift'], 3),
                })

        if pair_recs:
            pair_df = pd.DataFrame(pair_recs).drop_duplicates().sort_values('Lift', ascending=False).head(10)
            st.dataframe(pair_df.style.background_gradient(subset=['Lift'], cmap='RdYlGn'),
                         use_container_width=True)
        else:
            st.info("Tidak ada pasangan rak A-B yang ditemukan dari rules yang ada.")

    # Download
    csv_shelf = shelf_df.to_csv(index=False)
    st.download_button("⬇️ Download Klasifikasi Rak (CSV)", csv_shelf,
                       "klasifikasi_rak.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════
#  TAB 8 — 🔮 PREDIKSI PERILAKU PEMBELIAN
# ═══════════════════════════════════════════════════════════════
with tab8:
    st.subheader("🔮 Prediksi Perilaku Pembelian Pelanggan")

    st.markdown("""
    <div class="info-box">
    <b>Prediksi Perilaku Pembelian</b> menggunakan association rules untuk memperkirakan item apa yang
    kemungkinan besar akan dibeli pelanggan berikutnya, berdasarkan isi keranjang saat ini.
    Probabilitas pembelian diestimasi dari nilai <b>confidence</b> rule yang paling relevan.
    <br><br>
    🔴 Sangat Tinggi ≥ 80% &nbsp;|&nbsp; 🟠 Tinggi ≥ 60% &nbsp;|&nbsp; 🟡 Sedang ≥ 40% &nbsp;|&nbsp; 🟢 Rendah &lt; 40%
    </div>
    """, unsafe_allow_html=True)

    if 'rules_df' not in st.session_state:
        st.info("⬅️ Jalankan mining terlebih dahulu di tab **Proses Mining**.")
        st.stop()

    rules_p = st.session_state['rules_df']
    fi_p    = st.session_state['frequent_itemsets']

    if rules_p.empty:
        st.warning("Tidak ada rules. Turunkan parameter threshold lalu jalankan ulang mining.")
        st.stop()

    all_items_p = set()
    for _, row in rules_p.iterrows():
        for it in row['antecedent'].split(','):
            all_items_p.add(it.strip())
        for it in row['consequent'].split(','):
            all_items_p.add(it.strip())
    all_items_p_sorted = sorted(all_items_p)

    st.markdown("#### 🧺 Masukkan Isi Keranjang Saat Ini")
    cart_items = st.multiselect(
        "Pilih item yang sedang dibawa pelanggan:",
        options=all_items_p_sorted,
        help="Pilih item yang sudah ada di keranjang. Sistem akan memprediksi item berikutnya."
    )

    top_n_pred = st.slider("Jumlah prediksi:", 3, 15, 5, key="pred_slider")

    if cart_items:
        pred_df = predict_next_purchase(cart_items, rules_p, fi_p, top_n=top_n_pred)

        if pred_df.empty:
            st.warning("Tidak ada prediksi untuk kombinasi item ini. Coba tambah item lain ke keranjang.")
        else:
            st.markdown(f"#### 🔮 Prediksi {len(pred_df)} Item Berikutnya")

            for _, row in pred_df.iterrows():
                prob = row['Probabilitas Beli (%)']
                bar_color = '#2ecc71' if prob >= 80 else '#f39c12' if prob >= 60 else '#e67e22' if prob >= 40 else '#e74c3c'
                bar_width  = int(prob)

                st.markdown(f"""
                <div class="pred-card">
                    <b style="color:#f39c12; font-size:1.05rem">🔮 {row['Item Prediksi']}</b>
                    &nbsp; {row['Kategori']}<br>
                    <div style="background:#0f3460; border-radius:4px; height:8px; margin:6px 0;">
                        <div style="background:{bar_color}; width:{bar_width}%; height:8px; border-radius:4px;"></div>
                    </div>
                    <span style="color:#a8b2d8; font-size:0.83rem">
                        Probabilitas: <b style="color:{bar_color}">{prob:.1f}%</b> &nbsp;|&nbsp;
                        Lift: <b>{row['Lift']:.3f}</b> &nbsp;|&nbsp;
                        Support: <b>{row['Support (%)']:.1f}%</b><br>
                        Dipicu oleh: <i>{row['Dipicu Oleh']}</i>
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 📊 Visualisasi Probabilitas Prediksi")
            fig_pred = px.bar(
                pred_df,
                x='Probabilitas Beli (%)',
                y='Item Prediksi',
                orientation='h',
                color='Probabilitas Beli (%)',
                color_continuous_scale='YlOrRd',
                hover_data=['Dipicu Oleh', 'Lift', 'Support (%)'],
                labels={'Item Prediksi': 'Item'}
            )
            fig_pred.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#a8b2d8', height=350,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=5, r=5, t=10, b=5)
            )
            fig_pred.update_xaxes(gridcolor='#1a4a7a', range=[0, 100])
            fig_pred.update_yaxes(gridcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pred, use_container_width=True)

            # Analisis pola perilaku
            st.markdown("---")
            st.markdown("#### 🧠 Analisis Pola Perilaku")
            avg_prob = pred_df['Probabilitas Beli (%)'].mean()
            max_item = pred_df.iloc[0]['Item Prediksi']
            max_prob = pred_df.iloc[0]['Probabilitas Beli (%)']

            col1, col2, col3 = st.columns(3)
            col1.metric("Item Paling Mungkin Dibeli", max_item)
            col2.metric("Probabilitas Tertinggi",      f"{max_prob:.1f}%")
            col3.metric("Rata-rata Probabilitas",      f"{avg_prob:.1f}%")

            tinggi = len(pred_df[pred_df['Probabilitas Beli (%)'] >= 60])
            if tinggi > 0:
                st.success(f"✅ Ada **{tinggi} item** dengan probabilitas tinggi (≥60%) yang sangat disarankan untuk ditawarkan ke pelanggan ini.")
            else:
                st.info("ℹ️ Semua prediksi di bawah 60%. Pertimbangkan untuk menambah data transaksi agar pola lebih kuat.")

            csv_pred = pred_df.to_csv(index=False)
            st.download_button("⬇️ Download Prediksi (CSV)", csv_pred,
                               "prediksi_perilaku.csv", "text/csv")

    else:
        st.info("Pilih item keranjang di atas untuk melihat prediksi pembelian berikutnya.")

        # Tampilkan insight umum dari rules
        st.markdown("---")
        st.markdown("#### 📌 Insight Perilaku Umum dari Data")
        if not rules_p.empty:
            top_rules_insight = rules_p.head(5)
            for _, row in top_rules_insight.iterrows():
                st.markdown(f"""
                <div class="pred-card">
                    Pelanggan yang membeli <b style="color:#f39c12">{row['antecedent']}</b>
                    cenderung juga membeli <b style="color:#2ecc71">{row['consequent']}</b>
                    dengan probabilitas <b>{row['confidence']*100:.1f}%</b>
                    (Lift: {row['lift']:.3f})
                </div>
                """, unsafe_allow_html=True)