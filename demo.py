import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample extracted risks dictionary for simulation
extracted_risks = {
    'Archroma': {
        'Reputational Risk': 'Negative coverage about product quality and compliance in major publications.',
        'Regulatory/Compliance Risk': 'Fined for non-compliance with wastewater standards in Vietnam.',
        'Geopolitical Risk': 'Operations impacted due to political tensions in South Asia.',
        'Environmental Risk': 'Criticism for high carbon emissions and unsustainable sourcing.',
        'Financial Risk': 'Losses reported due to halted production from protests.'
    },
    'Huntsman': {
        'Reputational Risk': 'Criticism for use of hazardous chemicals in dyeing processes.',
        'Regulatory/Compliance Risk': 'Warning issued over REACH non-compliance in the EU.',
        'Geopolitical Risk': 'Facility closures in unstable regions.',
        'Environmental Risk': 'Cited in reports for water pollution.',
        'Financial Risk': 'Losses due to disruptions in China and Southeast Asia.'
    }
}

# Risk direction mapping (simplified logic)
risk_direction_mapping = {
    'Reputational Risk': 'Negative',
    'Regulatory/Compliance Risk': 'Negative',
    'Geopolitical Risk': 'Negative',
    'Environmental Risk': 'Negative',
    'Financial Risk': 'Negative'
}

# Function to generate table dynamically based on company
def generate_risk_table(company_name):
    if company_name not in extracted_risks:
        return pd.DataFrame(columns=["Risk Type", "Risk Direction", "Example"])

    company_risks = extracted_risks[company_name]
    data = {
        "Risk Type": list(company_risks.keys()),
        "Risk Direction": [risk_direction_mapping.get(rt, "Unknown") for rt in company_risks],
        "Example": list(company_risks.values())
    }
    return pd.DataFrame(data)

# Streamlit UI
st.title("Textile Supplier Risk Dashboard")

st.markdown("""
This dashboard allows you to analyze and visualize different categories of risks associated with major textile dye suppliers. 
Select a company from the dropdown to view the risk types, their examples, and various visual breakdowns.
""")

company_name = st.selectbox("Select a Company", list(extracted_risks.keys()))
df = generate_risk_table(company_name)

st.markdown(f"**Total Risks Identified for {company_name}: {len(df)}**")

st.subheader("Extracted Risk Table")
st.dataframe(df)

# Optional full descriptions toggle
if st.checkbox("Show Full Risk Descriptions"):
    st.markdown("### Detailed Risk Descriptions")
    for idx, row in df.iterrows():
        st.markdown(f"**{row['Risk Type']}** — {row['Example']}")

# Pie chart: risk type distribution
st.subheader("Risk Type Distribution")
fig1, ax1 = plt.subplots()
ax1.pie(df["Risk Type"].value_counts(), labels=df["Risk Type"].unique(), autopct='%1.1f%%')
ax1.axis('equal')
ax1.set_title("Distribution of Risk Types (All Risks Counted Once)")
st.pyplot(fig1)

# Additional Visualizations
st.subheader("Additional Risk Insights")

# Bar plot of risk types sorted by count
st.markdown("**Count of Risk Types**")
fig2, ax2 = plt.subplots()
sns.countplot(y="Risk Type", data=df, palette="viridis", ax=ax2,
              order=df["Risk Type"].value_counts().index)
ax2.set_title("Risk Type Count")
st.pyplot(fig2)

# Heatmap-like risk vs direction visual
st.markdown("**Risk Type and Direction Relationship**")
fig3, ax3 = plt.subplots()
ct = pd.crosstab(df["Risk Type"], df["Risk Direction"])
sns.heatmap(ct, annot=True, cmap="Reds", fmt="d", ax=ax3)
ax3.set_title("Heatmap of Risk Type vs Direction")
ax3.set_xlabel("Risk Direction")
ax3.set_ylabel("Risk Type")
st.pyplot(fig3)

# Horizontal bar with example text lengths (proxy for severity detail)
st.markdown("**Risk Detail (Text Length Proxy)**")
df["Example Length"] = df["Example"].apply(len)
fig4, ax4 = plt.subplots()
df_sorted = df.sort_values("Example Length")
ax4.barh(df_sorted["Risk Type"], df_sorted["Example Length"], color='orange')
ax4.set_xlabel("Length of Description")
ax4.set_title("Risk Severity Representation by Description Length")
st.pyplot(fig4)
