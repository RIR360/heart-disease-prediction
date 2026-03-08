import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Streamlit Demo", page_icon="📊")

st.title("📊 My GitHub-Hosted App")
st.markdown("""
This app was built and deployed directly from GitHub! 
Try moving the slider or changing the chart type below.
""")

# Sidebar settings
st.sidebar.header("Settings")
data_size = st.sidebar.slider("Number of rows", 10, 500, 100)
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line", "Bar", "Area"])

# Generate random data
data = pd.DataFrame(
    np.random.randn(data_size, 3),
    columns=['Category A', 'Category B', 'Category C']
)

# Display data and chart
st.subheader("Live Data Preview")
st.dataframe(data.head())

st.subheader(f"Visualization: {chart_type} Chart")
if chart_type == "Line":
    st.line_chart(data)
elif chart_type == "Bar":
    st.bar_chart(data)
else:
    st.area_chart(data)

if st.button("Celebrate!"):
    st.balloons()
    st.success("App updated successfully!")
