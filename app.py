import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(page_title="Retail Industry Dashboard", layout="wide")

# Title
st.title("Retail & Footwear Industry Dashboard")

# Load JSON data
@st.cache_data
def load_data():
    try:
        with open("Raw Data v4.json", "r") as f:
            data = json.load(f)
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Identify date column (try common variations)
        date_column = None
        possible_date_cols = ["date", "Date", "published_date", "publication_date", "published_datetime_utc"]
        for col in possible_date_cols:
            if col in df.columns:
                date_column = col
                break
        
        # Convert date column to datetime if found
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            # Rename to 'date' for consistency
            if date_column != "date":
                df = df.rename(columns={date_column: "date"})
        else:
            st.warning("No date column found in JSON. Date filtering will be disabled.")
        
        return df
    except FileNotFoundError:
        st.error("articles.json not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error("Invalid JSON format in articles.json.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Debug: Show DataFrame columns and sample data
with st.expander("Debug: JSON Data Structure"):
    st.write("Columns in DataFrame:", df.columns.tolist())
    st.write("Sample data (first 2 rows):")
    st.dataframe(df.head(2))

# Sidebar for filters
st.sidebar.header("Filters")
company_filter = st.sidebar.multiselect(
    "Select Company",
    options=["White Stuff", "Caleres", "TFG", "Rothy’s", "All"],
    default=["All"]
)

# Date filter (only if date column exists)
if "date" in df.columns:
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df["date"].min().date(), df["date"].max().date()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date()
    )
else:
    date_range = None
    st.sidebar.info("Date filtering disabled due to missing date column.")

# Filter data
filtered_df = df.copy()
if "All" not in company_filter and company_filter:
    # Assume 'content' or 'title' contains company info
    content_col = "content" if "content" in df.columns else "title" if "title" in df.columns else None
    if content_col:
        filtered_df = filtered_df[filtered_df[content_col].str.contains("|".join(company_filter), case=False, na=False)]
    else:
        st.warning("No 'content' or 'title' column found for company filtering.")

if date_range and "date" in df.columns:
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["date"] <= pd.to_datetime(date_range[1]))
    ]

# Extract key metrics (hard-coded based on provided summary)
metrics = {
    "White Stuff": {
        "Revenue (2024)": 154.8,  # £M
        "EBITDA (2024)": 8.6,    # £M
        "Sales Growth (Q4 2024)": 21.8,  # %
        "Full-Price Sales Growth": 26.9  # %
    },
    "Caleres": {
        "Net Sales Q2 2024": 683.3,  # $M
        "Net Earnings Q2 2024": 29.958,  # $M
        "Net Debt": 94.7,  # $M
        "Debt-to-EBITDA": 0.6
    }
}

# Display metrics
st.header("Key Metrics")
col1, col2 = st.columns(2)
for company, data in metrics.items():
    with col1 if company == "White Stuff" else col2:
        st.subheader(company)
        for key, value in data.items():
            st.metric(key, f"{value} {'£M' if company == 'White Stuff' and 'Growth' not in key else '$M' if company == 'Caleres' and 'Debt-to-EBITDA' not in key else ''}")

# Visualizations
st.header("Visualizations")

# Bar chart for sales/revenue
st.subheader("Revenue/Sales Comparison")
sales_data = pd.DataFrame({
    "Company": ["White Stuff", "Caleres"],
    "Value": [154.8, 683.3],  # £M for White Stuff, $M for Caleres
    "Metric": ["Revenue (2024)", "Net Sales Q2 2024"]
})
fig_sales = px.bar(sales_data, x="Company", y="Value", color="Metric", barmode="group",
                   title="Revenue/Sales Comparison",
                   labels={"Value": "Amount (in respective currencies)"})
st.plotly_chart(fig_sales, use_container_width=True)

# Pie chart for White Stuff revenue breakdown
st.subheader("White Stuff Revenue Breakdown (2024)")
revenue_breakdown = pd.DataFrame({
    "Channel": ["Stores & Online", "International & Wholesale"],
    "Revenue": [131.58, 23.22]  # £M, based on 85% and 15%
})
fig_pie = px.pie(revenue_breakdown, values="Revenue", names="Channel",
                 title="White Stuff Revenue by Channel")
st.plotly_chart(fig_pie, use_container_width=True)

# Line chart for White Stuff sales growth
st.subheader("White Stuff Sales Growth Trend")
growth_data = pd.DataFrame({
    "Period": ["2023", "Q4 2024"],
    "Sales Growth (%)": [13, 21.8]
})
fig_growth = px.line(growth_data, x="Period", y="Sales Growth (%)",
                     title="White Stuff Sales Growth Trend",
                     markers=True)
st.plotly_chart(fig_growth, use_container_width=True)

# Data table
st.header("Filtered Articles")
# Select columns that exist
available_cols = [col for col in ["title", "source", "date", "content"] if col in filtered_df.columns]
if available_cols:
    st.dataframe(filtered_df[available_cols].reset_index(drop=True))
else:
    st.warning("No displayable columns (title, source, date, content) found in JSON.")

# Instructions
st.markdown("""
### How to Use
- **Filters**: Use the sidebar to filter by company or date range (if available).
- **Metrics**: View key financial metrics for White Stuff and Caleres.
- **Visualizations**: Explore revenue, sales growth, and revenue breakdown charts.
- **Data Table**: Browse filtered articles with available fields.
- **Debug**: Check the 'Debug' expander to see JSON structure and sample data.
""")