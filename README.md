# Textile Dye Supplier Risk Analysis Dashboard

![Textile Risk Dashboard](https://img.shields.io/badge/Streamlit-App-blue?style=for-the-badge&logo=streamlit)

A Streamlit-based web application for analyzing and visualizing risks associated with top textile dye suppliers. This dashboard provides insights into risk trends, sentiment analysis, and actionable recommendations for the years 2023-2024.

## 📋 Overview

This project is designed to help stakeholders in the textile industry monitor and analyze risks related to dye suppliers. It integrates advanced risk classification, sentiment analysis using VADER, and AI-generated recommendations powered by Cohere. The dashboard offers a modern, dark-themed UI with interactive visualizations built using Plotly.

## ✨ Features

- **Supplier and Risk Filtering**: Select specific suppliers and risk categories to focus on relevant data.
- **Risk Classification**: Automatically categorizes risks into predefined types such as Geopolitical, Environmental, Financial, Supply Chain, and Market risks.
- **Sentiment Analysis**: Uses VADER to determine the risk direction (Positive, Negative, Neutral) based on article content.
- **AI-Powered Summaries and Recommendations**: Leverages Cohere API to summarize articles and provide actionable risk mitigation strategies.
- **Interactive Visualizations**:
  - Monthly risk trends over time.
  - Distribution of risk categories.
  - Average sentiment scores by risk category.
- **Model Performance Report**: Displays a classification report for risk category predictions.
- **Modern UI**: Dark theme with custom CSS, card layouts, and a responsive design for an enhanced user experience.

## 🛠️ Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/textile-risk-dashboard.git
   cd textile-risk-dashboard
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key Setup**:
   Replace the placeholder Cohere API key in `finalapp.py` with your own key. You can obtain an API key from [Cohere](https://cohere.ai/).

4. **Run the Application**:
   ```bash
   streamlit run finalapp.py
   ```
   The app will open in your default web browser at `http://localhost:8501`.

## 📊 Data

The dashboard uses a JSON dataset (`RawDatav4.json`) containing articles from 2023-2024 related to top textile dye suppliers. The data is filtered and processed to extract relevant risk information.

## 🔧 Technologies Used

- **Streamlit**: For building the interactive web application.
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For creating interactive charts and visualizations.
- **Cohere API**: For generating article summaries and recommendations.
- **NLTK (VADER)**: For sentiment analysis to determine risk direction.
- **Scikit-learn**: For potential model training and classification reporting.

## 📈 Visualizations

The dashboard includes several visualizations to help understand risk patterns:
- **Risk Direction Distribution**: A table showing the count of positive, negative, and neutral risks by category.
- **Monthly Risk Trends**: Line chart showing risk occurrences over time.
- **Sentiment Score by Risk Category**: Bar chart displaying average sentiment scores for each risk type.

## 🖼️ UI Design

The application features a dark theme with custom CSS for a modern look:
- **Responsive Layout**: Optimized for various screen sizes using Streamlit's wide layout.
- **Interactive Elements**: Hover effects on buttons and a clean, card-based design for content sections.
- **Footer**: A fixed footer with branding and copyright information.

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request. Ensure to follow the coding style and include appropriate documentation for any changes.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any inquiries or support, please reach out via GitHub Issues.

---

*Textile Risk Dashboard &copy; 2024 - Built with ❤️ using Streamlit*
