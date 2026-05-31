"""Amazon Review Intelligence — interactive Streamlit dashboard.

Loads the precomputed `data/reviews_dashboard.parquet` (built by
build_dashboard_data.py) plus the saved models, and exposes 5 tabs:

  1. Overview              — KPIs + rating / sentiment distributions
  2. Topics & Pain Points  — topic x sentiment heatmap, most negative topics
  3. Trends                — review volume & sentiment over time
  4. Mismatch Explorer     — rating vs model-predicted sentiment disagreements
  5. Live Predictor        — type a review, get sentiment + confidence + topic

Run:  streamlit run app.py
"""

import os
import warnings

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import distilbert_utils
import nlp_utils

warnings.filterwarnings("ignore")

DATA_PATH = "data/reviews_dashboard.parquet"
LDA_MODEL = "models/lda_model.joblib"
TOPIC_VECTORIZER = "models/topic_vectorizer.joblib"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "positive": "#2ca02c",
}

TOPIC_LABELS = {
    0: "Ease of Use & Cleaning",
    1: "Bedding & Sleep",
    2: "Bath & Soft Furnishings",
    3: "Home Appliances (Fans & Vacuums)",
    4: "Price, Value & Delivery Issues",
    5: "Furniture & Cutlery",
    6: "Size, Fit & Storage",
    7: "Cookware & Kitchen",
    8: "Drinkware & Beverages",
    9: "Gifts & Recommendations",
    10: "Appearance & Design",
    11: "Complaints & Malfunctions",
}

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
)

# Widen the sidebar so the bilingual topic chips aren't truncated ("Bath & Soft
# Furnish…"). The default ~21rem is too narrow once each label carries both languages.
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { min-width: 360px; max-width: 360px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Loaders (cached)
@st.cache_data(show_spinner="Loading reviews…")
def load_data(data_mtime):
    # data_mtime is the parquet's modification time; passing it as an argument
    # makes the cache invalidate automatically when build_dashboard_data.py
    # regenerates the file (otherwise stale data persists until server restart).
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    lda = joblib.load(LDA_MODEL)
    vectorizer = joblib.load(TOPIC_VECTORIZER)
    distilbert_utils.load()  # warm up the transformer once
    return lda, vectorizer


# Sidebar filters
def sidebar_filters(df):
    st.sidebar.header("Filters / Фильтры")

    min_d, max_d = df["timestamp"].min().date(), df["timestamp"].max().date()
    date_range = st.sidebar.slider(
        "Review date range / Период отзывов",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
    )

    topics = sorted(df["topic_label"].dropna().unique())
    chosen_topics = st.sidebar.multiselect("Topics / Темы", topics, default=topics)

    verified = st.sidebar.radio(
        "Verified purchase / Подтверждённая покупка",
        ["All / Все", "Verified only / Только подтверждённые", "Not verified / Без подтверждения"],
        index=0,
    )

    mask = (
        (df["timestamp"].dt.date >= date_range[0])
        & (df["timestamp"].dt.date <= date_range[1])
        & (df["topic_label"].isin(chosen_topics))
    )
    if verified.startswith("Verified only"):
        mask &= df["verified_purchase"]
    elif verified.startswith("Not verified"):
        mask &= ~df["verified_purchase"]

    st.sidebar.caption(
        f"{int(mask.sum()):,} of {len(df):,} reviews selected / "
        f"выбрано из {len(df):,}"
    )
    return df[mask]


# Tab 1 — Overview
def tab_overview(df):
    st.subheader("Overview / Обзор")

    if df.empty:
        st.info(
            "No reviews match the current filters. / "
            "Нет отзывов под текущие фильтры."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews / Отзывы", f"{len(df):,}")
    c2.metric("Avg rating / Ср. рейтинг", f"{df['rating'].mean():.2f} ★")
    c3.metric("% Positive / % позитива", f"{(df['sentiment_label'] == 'positive').mean() * 100:.1f}%")
    c4.metric("% Verified / % подтверждённых", f"{df['verified_purchase'].mean() * 100:.1f}%")

    col1, col2 = st.columns(2)

    rating_counts = df["rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["rating", "count"]
    fig_r = px.bar(
        rating_counts,
        x="rating",
        y="count",
        title="Rating distribution / Распределение оценок",
        text_auto=".2s",
    )
    fig_r.update_layout(
        xaxis_title="Star rating / Оценка (звёзды)", yaxis_title="Reviews / Отзывы"
    )
    col1.plotly_chart(fig_r, width='stretch')

    sent_counts = (
        df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER).reset_index()
    )
    sent_counts.columns = ["sentiment", "count"]
    fig_s = px.bar(
        sent_counts,
        x="sentiment",
        y="count",
        title="Sentiment distribution (rating-based) / Тональность (по рейтингу)",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        text_auto=".2s",
    )
    fig_s.update_layout(showlegend=False, xaxis_title="", yaxis_title="Reviews / Отзывы")
    col2.plotly_chart(fig_s, width='stretch')

    st.caption(
        "Sentiment here is derived from the star rating (1–2 negative, 3 neutral, "
        "4–5 positive) — the ground-truth label used to train the models. / "
        "Тональность выведена из звёздного рейтинга (1–2 negative, 3 neutral, "
        "4–5 positive) — это ground-truth метка, на которой обучались модели."
    )


# Tab 2 — Topics & Pain Points
def tab_topics(df):
    st.subheader("Topics & Pain Points / Темы и проблемы")

    if df.empty:
        st.info(
            "No reviews match the current filters. / "
            "Нет отзывов под текущие фильтры."
        )
        return

    # Topic x sentiment heatmap (row-normalized)
    ct = pd.crosstab(df["topic_label"], df["sentiment_label"], normalize="index")
    ct = ct.reindex(columns=SENTIMENT_ORDER).fillna(0)
    ct = ct.sort_values("negative", ascending=False)
    fig_h = px.imshow(
        (ct * 100).round(1),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="% of topic / % темы"),
        title="Sentiment mix per topic (%) / Тональность по темам (%)",
    )
    fig_h.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_h, width='stretch')

    col1, col2 = st.columns(2)

    neg_share = (
        df.assign(is_neg=df["sentiment_label"] == "negative")
        .groupby("topic_label")["is_neg"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    neg_share["is_neg"] = (neg_share["is_neg"] * 100).round(1)
    fig_neg = px.bar(
        neg_share,
        x="is_neg",
        y="topic_label",
        orientation="h",
        title="Most negative topics (% negative reviews) / Самые негативные темы (% негатива)",
        color="is_neg",
        color_continuous_scale="Reds",
    )
    fig_neg.update_layout(
        yaxis_title="", xaxis_title="% negative / % негатива", coloraxis_showscale=False
    )
    fig_neg.update_yaxes(categoryorder="total ascending")
    col1.plotly_chart(fig_neg, width='stretch')

    helpful = (
        df.groupby("topic_label")["helpful_vote"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    helpful["helpful_vote"] = helpful["helpful_vote"].round(2)
    fig_help = px.bar(
        helpful,
        x="helpful_vote",
        y="topic_label",
        orientation="h",
        title="Avg helpful votes per topic / Ср. число «полезно» по темам",
        color="helpful_vote",
        color_continuous_scale="Blues",
    )
    fig_help.update_layout(
        yaxis_title="", xaxis_title="Avg helpful votes / Ср. «полезно»", coloraxis_showscale=False
    )
    fig_help.update_yaxes(categoryorder="total ascending")
    col2.plotly_chart(fig_help, width='stretch')

    st.markdown("#### Sample reviews by topic / Примеры отзывов по теме")
    pick = st.selectbox(
        "Choose a topic / Выберите тему", sorted(df["topic_label"].dropna().unique())
    )
    sample = df[df["topic_label"] == pick][
        ["rating", "sentiment_label", "title", "text", "helpful_vote"]
    ].head(200)
    st.dataframe(
        sample.sample(min(10, len(sample)), random_state=42),
        width='stretch',
        hide_index=True,
    )


# Tab 3 — Trends
def tab_trends(df):
    st.subheader("Trends over time / Динамика во времени")

    if df.empty:
        st.info(
            "No reviews match the current filters. / "
            "Нет отзывов под текущие фильтры."
        )
        return

    volume = df.groupby("date").size().reset_index(name="count")
    fig_v = px.line(
        volume,
        x="date",
        y="count",
        title="Review volume per month / Объём отзывов по месяцам",
    )
    fig_v.update_layout(xaxis_title="", yaxis_title="Reviews / Отзывы")
    st.plotly_chart(fig_v, width='stretch')

    sent_time = (
        pd.crosstab(df["date"], df["sentiment_label"], normalize="index")
        .reindex(columns=SENTIMENT_ORDER)
        .fillna(0)
        * 100
    ).reset_index()
    sent_long = sent_time.melt(
        id_vars="date", var_name="sentiment", value_name="share"
    )
    fig_st = px.line(
        sent_long,
        x="date",
        y="share",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment share over time (%) / Доли тональности во времени (%)",
    )
    fig_st.update_layout(xaxis_title="", yaxis_title="% of reviews / % отзывов")
    st.plotly_chart(fig_st, width='stretch')


# Tab 4 — Mismatch Explorer
def tab_mismatch(df):
    st.subheader("Sentiment–Rating Mismatch / Расхождение тональности и рейтинга")
    st.caption(
        "Reviews where the model's predicted sentiment disagrees with the star "
        "rating. Confidence is the DistilBERT softmax score — a rough filter, not "
        "a calibrated probability (transformer softmax tends to be overconfident), "
        "so raising the threshold trims the lowest-certainty cases but the cutoff "
        "value is not directly comparable to the calibrated LinearSVC thresholds in "
        "the notebook. / Отзывы, где прогноз модели расходится со звёздным рейтингом. "
        "Confidence — это softmax DistilBERT: грубый фильтр, а не калиброванная "
        "вероятность (softmax трансформера склонен к переуверенности), поэтому "
        "повышение порога лишь отсекает наименее уверенные случаи и не сопоставимо "
        "напрямую с калиброванными порогами LinearSVC из ноутбука."
    )

    if df.empty:
        st.info(
            "No reviews match the current filters. / "
            "Нет отзывов под текущие фильтры."
        )
        return

    threshold = st.slider(
        "Min model confidence / Мин. уверенность модели", 0.5, 1.0, 0.9, 0.01
    )

    five_neg = df[
        (df["rating"] == 5)
        & (df["predicted_label"] == "negative")
        & (df["confidence"] >= threshold)
    ]
    one_pos = df[
        (df["rating"] == 1)
        & (df["predicted_label"] == "positive")
        & (df["confidence"] >= threshold)
    ]

    c1, c2 = st.columns(2)
    c1.metric("⭐5 but predicted negative / ⭐5, прогноз negative", f"{len(five_neg):,}")
    c2.metric("⭐1 but predicted positive / ⭐1, прогноз positive", f"{len(one_pos):,}")

    cols = ["rating", "predicted_label", "confidence", "title", "text", "topic_label"]

    st.markdown(
        "#### ⭐5 rating, negative text — hidden dissatisfaction / "
        "⭐5, но негативный текст — скрытое недовольство"
    )
    st.dataframe(
        five_neg.sort_values("confidence", ascending=False)[cols].head(15),
        width='stretch',
        hide_index=True,
    )

    st.markdown(
        "#### ⭐1 rating, positive text — possible mis-rating / "
        "⭐1, но позитивный текст — возможная ошибка в оценке"
    )
    st.dataframe(
        one_pos.sort_values("confidence", ascending=False)[cols].head(15),
        width='stretch',
        hide_index=True,
    )


# Tab 5 — Live Predictor
def tab_predictor(models):
    st.subheader("Live Review Predictor / Прогноз по отзыву")
    st.caption(
        "Type a review below. The fine-tuned DistilBERT model predicts sentiment "
        "and confidence; LDA assigns the most likely topic. / Введите отзыв ниже. "
        "Дообученный DistilBERT предскажет тональность и уверенность, а LDA — "
        "наиболее вероятную тему."
    )

    lda, vectorizer = models

    text = st.text_area(
        "Review text / Текст отзыва",
        value="The handle broke after two uses and the seller refused a refund.",
        height=120,
    )

    if st.button("Analyze / Анализировать", type="primary"):
        if not text.strip():
            st.warning(
                "Please enter some review text to analyze. / "
                "Введите текст отзыва для анализа."
            )
        else:
            proba = distilbert_utils.predict_proba([text])[0]
            classes = distilbert_utils.label_list()
            pred = classes[int(proba.argmax())]
            cleaned = nlp_utils.remove_stopwords(nlp_utils.clean_review_text(text))
            if cleaned:
                topic_id = int(
                    lda.transform(vectorizer.transform([cleaned])).argmax(axis=1)[0]
                )
                topic_name = TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")
            else:
                topic_name = None  # nothing meaningful left after cleaning

            # Persist in session_state so the result survives reruns (e.g. tab
            # switches) instead of vanishing the moment the button resets to False.
            st.session_state["prediction"] = {
                "pred": pred,
                "confidence": float(proba.max()),
                "classes": classes,
                "proba": proba.tolist(),
                "topic_name": topic_name,
            }

    result = st.session_state.get("prediction")
    if result:
        pred = result["pred"]
        color = {"negative": "red", "positive": "green"}.get(pred, "gray")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Sentiment / Тональность: :{color}[{pred.upper()}]")
            st.metric("Confidence / Уверенность", f"{result['confidence'] * 100:.1f}%")
            if result["topic_name"]:
                st.markdown(f"**Predicted topic / Тема:** {result['topic_name']}")
            else:
                st.caption(
                    "Topic unavailable — text too short after cleaning. / "
                    "Тема недоступна — текст слишком короткий после очистки."
                )
        with c2:
            proba_df = pd.DataFrame(
                {"sentiment": result["classes"], "probability": result["proba"]}
            )
            fig = px.bar(
                proba_df,
                x="sentiment",
                y="probability",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Class probabilities / Вероятности классов",
                range_y=[0, 1],
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width='stretch')


# Main
def main():
    st.title("📊 Amazon Review Intelligence")
    st.caption(
        "NLP analytics on Amazon Home & Kitchen reviews / "
        "NLP-аналитика отзывов Amazon Home & Kitchen"
    )

    df = load_data(os.path.getmtime(DATA_PATH))
    models = load_models()
    filtered = sidebar_filters(df)

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "📊 Overview / Обзор",
            "🏷️ Topics & Pain Points / Темы и проблемы",
            "📈 Trends / Динамика",
            "⚠️ Mismatch Explorer / Расхождения",
            "🔮 Live Predictor / Прогноз",
        ]
    )
    with t1:
        tab_overview(filtered)
    with t2:
        tab_topics(filtered)
    with t3:
        tab_trends(filtered)
    with t4:
        tab_mismatch(filtered)
    with t5:
        tab_predictor(models)


if __name__ == "__main__":
    main()
