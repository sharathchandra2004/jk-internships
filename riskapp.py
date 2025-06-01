import streamlit as st
import pandas as pd
import json
import cohere
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

COHERE_API_KEY = "HNrggtGmhoC2Ewkitr7qAVhpuOcIo1x91Tukl2f1"  # Replace with your real API key
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

def classify_risk(text):
    text = text.lower()
    for risk, keywords in risk_keywords.items():
        if any(k in text for k in keywords):
            if any(p in text for p in ["growth", "improve", "expand", "profit"]):
                return risk, "Positive (Decreased Risk)"
            elif any(n in text for n in ["loss", "issue", "decline", "disruption"]):
                return risk, "Negative (Increased Risk)"
            else:
                return risk, "Neutral"
    return None, None

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
    df['risk_category'], df['risk_direction'] = zip(*df['processed_text'].apply(classify_risk))
    df['summary'] = df['Full_Article'].apply(full_summary)
    df['date'] = pd.to_datetime(df['published_datetime_utc'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

def recommend_actions(risk, direction):
    if direction == "Negative (Increased Risk)":
        if risk == "Financial and Operational Risks":
            return "- Conduct regular audits\n- Improve liquidity\n- Review vendor performance"
        if risk == "Market and Competitive Risks":
            return "- Monitor competitors\n- Diversify markets\n- Lock key contracts"
        if risk == "Supply Chain and Logistics Risks":
            return "- Diversify logistics routes\n- Digitize SCM tools\n- Keep buffer stock"
        if risk == "Geopolitical and Regulatory Risks":
            return "- Track trade policy\n- Diversify supplier geography\n- Comply with evolving rules"
        if risk == "Agricultural and Environmental Risks":
            return "- Use eco-friendly materials\n- Engage in reforestation/sustainability\n- Monitor weather impact"
    elif direction == "Positive (Decreased Risk)":
        return "- Maintain current risk posture\n- Strengthen successful practices"
    else:
        return "- Observe trends closely\n- Prepare response strategies"

st.title("📊 Textile Dye Supplier Risk Analysis (2023–2024)")
df = load_data()

# Supplier selection (top 10)
selected_supplier = st.selectbox("Select a Supplier", ["All"] + suppliers, key="supplier_select")
selected_risk = st.selectbox("Select a Risk Category", risk_categories, key="risk_select")


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
        st.markdown(f"**Recommendations**:\n{recommend_actions(row['risk_category'], row['risk_direction'])}")
        
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

    # 2. Risk Exposure by Supplier and Risk Category (Grouped Bar Chart)
    heatmap_df = df.copy()
    if selected_supplier != "All":
        heatmap_df = heatmap_df[heatmap_df['supplier'] == selected_supplier]
    if selected_risk != "All":
        heatmap_df = heatmap_df[heatmap_df['risk_category'] == selected_risk]

    if not heatmap_df.empty:
        grouped = heatmap_df.groupby(['supplier', 'risk_category']).size().reset_index(name='Count')
        if not grouped.empty:
            fig_grouped = px.bar(grouped, x='supplier', y='Count', color='risk_category', barmode='group',
                                 title="📊 Risk Exposure by Supplier and Category")
            st.plotly_chart(fig_grouped, use_container_width=True)
    else:
        st.info("No data for risk exposure chart.")

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
