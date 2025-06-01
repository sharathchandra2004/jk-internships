import streamlit as st
import pandas as pd
import plotly.express as px

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .risk-high { color: red; font-weight: bold; }
        .risk-medium { color: orange; font-weight: bold; }
        .risk-low { color: green; font-weight: bold; }
        .article-box {
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .header-text {
            font-size: 30px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Risk Intelligence Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("📊 Filters")
selected_supplier = st.sidebar.selectbox("Select Supplier:", options=["All"] + sorted(df['Supplier'].unique()))
selected_year = st.sidebar.selectbox("Select Year:", options=["All"] + sorted(df['Year'].astype(str).unique()))
selected_category = st.sidebar.selectbox("Select Risk Category:", options=["All"] + sorted(df['Risk_Category'].unique()))

# --- Filter logic ---
filtered_df = df.copy()
if selected_supplier != "All":
    filtered_df = filtered_df[filtered_df['Supplier'] == selected_supplier]
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['Year'].astype(str) == selected_year]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_category]

# --- Summary Counts ---
st.subheader("📌 Risk Summary")
summary = filtered_df.groupby(['Risk_Category']).size().reset_index(name='Count')
st.dataframe(summary, use_container_width=True)

# --- Risk Category Pie Chart ---
if not summary.empty:
    pie_fig = px.pie(summary, values='Count', names='Risk_Category', title='Risk Category Distribution')
    st.plotly_chart(pie_fig, use_container_width=True)

# --- Risk Trends Over Time ---
st.subheader("📈 Risk Trends Over Time")
if not filtered_df.empty:
    trend_df = filtered_df.groupby(['Year', 'Risk_Category']).size().reset_index(name='Count')
    line_fig = px.line(trend_df, x='Year', y='Count', color='Risk_Category', markers=True)
    st.plotly_chart(line_fig, use_container_width=True)

# --- Display Articles in Expanders ---
st.subheader("📰 Detailed Risk Articles")
for _, row in filtered_df.iterrows():
    risk_color = 'risk-high' if row['Risk_Level'] == 'High' else 'risk-medium' if row['Risk_Level'] == 'Medium' else 'risk-low'
    with st.expander(f"{row['Article_Title'][:80]}..."):
        st.markdown(f"""
            <div class='article-box'>
                <p><strong>Supplier:</strong> {row['Supplier']} | <strong>Year:</strong> {row['Year']}</p>
                <p><strong>Category:</strong> {row['Risk_Category']} | <strong>Level:</strong> <span class='{risk_color}'>{row['Risk_Level']}</span></p>
                <p><strong>Direction:</strong> {row['Risk_Direction']}</p>
                <p><strong>Summary:</strong> {row['Article_Summary']}</p>
                <p><strong>Recommendation:</strong> {row['Recommendation']}</p>
                <p><a href='{row['URL']}' target='_blank'>🔗 Read full article</a></p>
            </div>
        """, unsafe_allow_html=True)

# --- End of app ---
st.markdown("---")
st.markdown("Developed by Sharath | Powered by Streamlit", unsafe_allow_html=True)
