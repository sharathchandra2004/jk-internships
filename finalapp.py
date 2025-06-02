import streamlit as st
import pandas as pd
import json
import cohere
import re
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

nltk.download('punkt', download_dir='nltk_data')
nltk.download('punkt_tab', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')
nltk.download('vader_lexicon', download_dir='nltk_data')
import nltk
import os

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)


COHERE_API_KEY = "3YcnwjTjyDNDUHJ3gF7Gu4F6Sc2t3e54Le8Zss64"  # Replace with your real API key
co = cohere.Client(COHERE_API_KEY)

# Suppliers and risks lists
suppliers = [
    "Welspun Living Limited", "Teejay Lanka PLC", "Arvind Limited", "Caleres, Inc.",
    "Interloop Limited", "Kitex Garments Limited", "ThredUp Inc.",
    "G-III Apparel Group, Ltd.", "Mint Velvet", "White Stuff Limited"
]
risk_categories = [
    "All",
    "Geopolitical and Regulatory Risks", "Agricultural and Environmental Risks",
    "Financial and Operational Risks", "Supply Chain and Logistics Risks",
    "Market and Competitive Risks"
]

risk_keywords = {
    "Geopolitical and Regulatory Risks": ["trade war", "tariff", "regulation", "policy", "sanction", "government", "political"],
    "Agricultural and Environmental Risks": ["drought", "climate change", "crop failure", "environment", "sustainability", "weather"],
    "Financial and Operational Risks": ["bankruptcy", "financial instability", "labor strike", "production issue", "cost", "debt", "profit"],
    "Supply Chain and Logistics Risks": ["transportation", "logistics", "supply chain", "disruption", "fuel price", "delivery"],
    "Market and Competitive Risks": ["competition", "price fluctuation", "market share", "competitor", "demand", "sales"]
}

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Fast keyword-based risk category classifier

def classify_risk(text, df=None):
    # 1. Risk Direction (sentiment)
    sentiment_scores = sentiment_analyzer.polarity_scores(text[:512])
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        direction = "Positive (Decreased Risk)"
    elif compound <= -0.05:
        direction = "Negative (Increased Risk)"
    else:
        direction = "Neutral"

    # 2. Risk Category (robust keyword match)
    best_category = None
    max_count = 0
    text_lower = text.lower()
    for category, keywords in risk_keywords.items():
        matches = set()
        for kw in keywords:
            kw_lower = kw.lower()
            # Substring match
            if kw_lower in text_lower:
                matches.add(kw)
            # Word boundary match for single words
            elif ' ' not in kw_lower and re.search(r'\\b' + re.escape(kw_lower) + r'\\b', text_lower):
                matches.add(kw)
        if len(matches) > max_count:
            max_count = len(matches)
            best_category = category
    if not best_category or max_count == 0:
        # Fallback: assign most common risk category in the dataset, or a default
        if df is not None and 'risk_category' in df.columns:
            fallback = df['risk_category'].value_counts().idxmax()
        else:
            fallback = "Market and Competitive Risks"
        best_category = fallback

    return best_category, direction

# Text preprocessor
def preprocess_text(text):
    if not text or text == "null":
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Reduced length summary using cohere summarize-xsmall (fallback to first 3 sentences)
def full_summary(text):
    if not text or not isinstance(text, str):
        return ""
    try:
        response = co.summarize(
            text=text,
            length="short",  # Use "medium" if you want slightly more detailed summaries
            format="paragraph",
            model="summarize-xsmall",
            additional_command=None,
            temperature=0.3,
        )
        return response.summary
    except Exception:
        # Fallback: return the first 3 complete sentences
        sentences = text.split('.')
        return '. '.join(sentences[:3]).strip()



# Filter for 2023-2024 articles only
def filter_by_date(data):
    result = []
    for article in data:
        try:
            date = datetime.strptime(article['published_datetime_utc'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if date.year in [2023, 2024]:
                result.append(article)
        except:
            continue
    return result

def identify_supplier(text, title):
    text = (text or "").lower()
    title = (title or "").lower()
    for s in suppliers:
        if s.lower() in text or s.lower() in title:
            return s
    return None

@st.cache_data(show_spinner=False)
def load_data():
    with open("RawDatav4.json", 'r') as f:
        raw = json.load(f)
    articles = filter_by_date(raw)
    df = pd.DataFrame(articles)
    df['supplier'] = df.apply(lambda row: identify_supplier(row.get('Full_Article', ''), row.get('title', '')), axis=1)
    df = df[df['supplier'].isin(suppliers)]  # limit to top 10 suppliers only
    df.dropna(subset=['supplier'], inplace=True)
    df['processed_text'] = df['Full_Article'].apply(preprocess_text)
    df['risk_category'], df['risk_direction'] = zip(*df['processed_text'].apply(lambda x: classify_risk(x, df)))
    df['summary'] = df['Full_Article'].apply(full_summary)
    df['date'] = pd.to_datetime(df['published_datetime_utc'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

# Model-based recommendation using Cohere

def generate_recommendation(article_text, summary=None):
    prompt = (
        "Given the following article about a textile supplier, recommend specific actions the supplier should take to mitigate any risks or issues mentioned. "
        "Be concise and actionable.\n\n"
    )
    if summary:
        prompt += f"Article Summary: {summary}\n"
    else:
        prompt += f"Article: {article_text}\n"
    prompt += "\nRecommendations:"
    try:
        response = co.chat(
            model="command-xlarge-nightly",
            message=prompt,
            max_tokens=100,
            temperature=0.5
        )
        return response.text.strip()
    except Exception as e:
        return "(Could not generate recommendations: " + str(e) + ")"

# Modern Streamlit page config and custom CSS
st.set_page_config(page_title="Textile Risk Dashboard", page_icon="🧵", layout="wide")

# Custom CSS for dark theme, fonts, and hover effects
st.markdown('''
    <style>
    html, body, [class*="css"]  {
        background-color: #18191A;
        color: #F5F6F7;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stApp {
        background-color: #18191A;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F5F6F7;
        font-weight: 700;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDataFrame, .stTable {
        background: #23272B !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .stDataFrame tbody tr:hover, .stTable tbody tr:hover {
        background: #31363B !important;
    }
    .stButton>button {
        color: #fff;
        background: linear-gradient(90deg, #4F8DFD 0%, #235390 100%);
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #235390 0%, #4F8DFD 100%);
    }
    .stSelectbox>div>div {
        background: #23272B !important;
        color: #F5F6F7 !important;
        border-radius: 6px;
    }
    footer {visibility: hidden;}
    .custom-footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        width: 100%;
        background: #23272B;
        color: #aaa;
        text-align: center;
        padding: 0.5rem 0;
        font-size: 0.95rem;
        z-index: 100;
    }
    </style>
''', unsafe_allow_html=True)

st.title("🧵 Textile Dye Supplier Risk Analysis (2023–2024)")
st.markdown("""
<div style='font-size:1.2rem; color:#B0B3B8; margin-bottom:1.5rem;'>
    An advanced dashboard for analyzing and visualizing risk trends, sentiment, and recommendations for top textile dye suppliers.
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data()

# Use columns for controls and summary metrics
col1, col2 = st.columns([2, 3])
with col1:
    selected_supplier = st.selectbox("🔎 Select a Supplier", ["All"] + suppliers, key="supplier_select")
with col2:
    selected_risk = st.selectbox("⚠️ Select a Risk Category", risk_categories, key="risk_select")

st.markdown("---")

# Add the Risk Direction Distribution table here
if selected_supplier != "All":
    st.subheader(f"Risk Direction Distribution for {selected_supplier}")
    supplier_data = df[df['supplier'] == selected_supplier]
    if not supplier_data.empty:
        risk_direction_counts = supplier_data.groupby(['risk_category', 'risk_direction']).size().unstack(fill_value=0).reset_index()
        for direction in ["Positive (Decreased Risk)", "Negative (Increased Risk)", "Neutral"]:
            if direction not in risk_direction_counts.columns:
                risk_direction_counts[direction] = 0
        risk_direction_table = pd.DataFrame(index=risk_categories[1:],  # exclude "All"
                                           columns=["Positive (Decreased Risk)", "Negative (Increased Risk)", "Neutral"]).fillna(0)
        for _, row in risk_direction_counts.iterrows():
            risk_category = row['risk_category']
            if risk_category in risk_direction_table.index:
                risk_direction_table.loc[risk_category, "Positive (Decreased Risk)"] = row.get("Positive (Decreased Risk)", 0)
                risk_direction_table.loc[risk_category, "Negative (Increased Risk)"] = row.get("Negative (Increased Risk)", 0)
                risk_direction_table.loc[risk_category, "Neutral"] = row.get("Neutral", 0)
        risk_direction_table.index.name = "Risk Category"
        st.write("Risk Direction Counts by Category:")
        st.table(risk_direction_table)
    else:
        st.write("No data available for this supplier to display the risk direction distribution.")



# Filter data based on selections
filtered_df = df.copy()
if selected_supplier != "All":
    filtered_df = filtered_df[filtered_df['supplier'] == selected_supplier]
if selected_risk != "All":
    filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk]

if filtered_df.empty:
    st.warning("⚠️ No articles found for the selected supplier and risk category.")
else:
    st.subheader("📌 Risk Summary")

    # # Pie chart showing distribution of risk categories for selected supplier and risk
    # pie_data = filtered_df['risk_category'].value_counts(normalize=True).reset_index()
    # pie_data.columns = ['Risk Category', 'Proportion']
    # fig = px.pie(pie_data, names='Risk Category', values='Proportion',
    #              title=f"Risk Distribution for Supplier: {selected_supplier}, Risk: {selected_risk}",
    #              hole=0.3)
    # st.plotly_chart(fig, use_container_width=True)

    st.subheader("📑 Article Summaries with Risk")
    for _, row in filtered_df.iterrows():
        st.markdown(f"**📰 {row['title']}**")
        st.write(f"**Date**: {row['published_datetime_utc'].split('T')[0]}")
        st.write(f"**Risk Category**: {row['risk_category']} | **Direction**: {row['risk_direction']}")
        st.write(f"**Summary**: {row['summary']}")
        # Use Cohere to generate recommendations
        recommendations = generate_recommendation(row['Full_Article'], row['summary'])
        st.markdown(f"**Recommendations**:\n{recommendations}")
        st.markdown("---")
        

    # Add additional charts section
    st.markdown("## 📈 Additional Risk Visualizations")

    # 1. Monthly Risk Trend Over Time
    trend_df = filtered_df.copy()
    if selected_risk != "All":
        trend_df = trend_df[trend_df['risk_category'] == selected_risk]
    trend_grouped = trend_df.groupby(['month', 'risk_category']).size().reset_index(name='Count')
    if not trend_grouped.empty:
        fig_trend = px.line(trend_grouped, x='month', y='Count', color='risk_category',
                            title='📅 Monthly Risk Trend by Category', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data for monthly trend.")

    # 5. Bar chart for average sentiment score by risk category
    st.markdown("### 📊 Average Sentiment Score by Risk Category")
    if 'processed_text' in filtered_df.columns and filtered_df['risk_category'].nunique() > 1:
        # Only use rows with valid processed_text
        valid_df = filtered_df[filtered_df['processed_text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        if not valid_df.empty:
            valid_df = valid_df.copy()
            valid_df['sentiment_score'] = valid_df['processed_text'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
            avg_sentiment = valid_df.groupby('risk_category')['sentiment_score'].mean().reset_index()
            avg_sentiment = avg_sentiment.sort_values('sentiment_score', ascending=False)
            try:
                fig_sentiment = px.bar(
                    avg_sentiment,
                    x='risk_category',
                    y='sentiment_score',
                    title='Average Sentiment Score by Risk Category',
                    labels={'risk_category': 'Risk Category', 'sentiment_score': 'Avg Sentiment Score'},
                    text_auto='.2f'
                )
                fig_sentiment.update_traces(textposition='outside')
            except TypeError:
                fig_sentiment = px.bar(
                    avg_sentiment,
                    x='risk_category',
                    y='sentiment_score',
                    title='Average Sentiment Score by Risk Category',
                    labels={'risk_category': 'Risk Category', 'sentiment_score': 'Avg Sentiment Score'}
                )
                fig_sentiment.update_traces(
                    text=[f"{v:.2f}" for v in avg_sentiment['sentiment_score']],
                    textposition='outside'
                )
            fig_sentiment.update_xaxes(tickangle=30, tickfont=dict(size=12))
            st.plotly_chart(fig_sentiment, use_container_width=True)
            st.dataframe(avg_sentiment)
        else:
            st.info("Not enough valid data for sentiment score chart.")
    else:
        st.info("Not enough data for sentiment score chart.")

    # === Add model classification report ===
    st.markdown("## 🧪 Model Classification Report")

    # For demo, replace these with your actual true labels and predictions arrays
    # Example dummy data:
    y_true = filtered_df['risk_category'].dropna()
    # Dummy predictions - here we just assume the model predicted 'Financial and Operational Risks' for all, replace with real predictions
    y_pred = ["Financial and Operational Risks"] * len(y_true)

    if len(y_true) > 0:
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Display classification report dataframe with some formatting
        st.dataframe(report_df.style.format({
            'precision': "{:.2f}",
            'recall': "{:.2f}",
            'f1-score': "{:.2f}",
            'support': "{:.0f}"
        }))
    else:
        st.info("Not enough data to generate classification report.")

# Custom footer
st.markdown("""
<div class='custom-footer'>
    Textile Risk Dashboard &copy; 2024 &mdash; Designed with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
