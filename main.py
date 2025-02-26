import streamlit as st
import pandas as pd
from src.data_generator import DataGenerator
from src.model import ClickPredictionModel
from src.visualization import DashboardVisualizer

# Page config
st.set_page_config(
    page_title="Click-through Rate Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    data_gen = DataGenerator(n_samples=1000)
    st.session_state.data = data_gen.generate_data()
    st.session_state.model = ClickPredictionModel()
    st.session_state.model.train(st.session_state.data)

# Sidebar
st.sidebar.title("Dashboard Controls")
sample_size = st.sidebar.slider("Sample Size", 100, 2000, 1000, 100)
if st.sidebar.button("Regenerate Data"):
    data_gen = DataGenerator(n_samples=sample_size)
    st.session_state.data = data_gen.generate_data()
    st.session_state.model.train(st.session_state.data)

# Main content
tab1, tab2 = st.tabs(["Analytics Dashboard", "Dataset View"])

with tab1:
    st.title("Click-through Rate Analytics Dashboard")
    
    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_ctr = st.session_state.data['clicked'].mean()
        st.metric("Average CTR", f"{avg_ctr:.2%}")
    with col2:
        avg_time = st.session_state.data['time_on_screen'].mean()
        st.metric("Avg Time on Screen", f"{avg_time:.1f}s")
    with col3:
        bounce_rate = st.session_state.data['exited_screen'].mean()
        st.metric("Bounce Rate", f"{bounce_rate:.2%}")
    
    # Feature importance plot
    st.subheader("Feature Importance Analysis")
    feature_importance_plot = DashboardVisualizer.create_feature_importance_plot(
        st.session_state.model.feature_importance
    )
    st.plotly_chart(feature_importance_plot, use_container_width=True)
    
    # Click-through rate by category
    col1, col2 = st.columns(2)
    with col1:
        device_plot = DashboardVisualizer.create_click_rate_by_category(
            st.session_state.data, 'device_type'
        )
        st.plotly_chart(device_plot, use_container_width=True)
    
    with col2:
        browser_plot = DashboardVisualizer.create_click_rate_by_category(
            st.session_state.data, 'browser'
        )
        st.plotly_chart(browser_plot, use_container_width=True)
    
    # Time series and correlation
    col1, col2 = st.columns(2)
    with col1:
        time_series = DashboardVisualizer.create_time_series_plot(st.session_state.data)
        st.plotly_chart(time_series, use_container_width=True)
    
    with col2:
        correlation = DashboardVisualizer.create_correlation_heatmap(st.session_state.data)
        st.plotly_chart(correlation, use_container_width=True)

with tab2:
    st.title("Dataset View")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        device_filter = st.multiselect(
            "Device Type",
            options=st.session_state.data['device_type'].unique()
        )
    with col2:
        browser_filter = st.multiselect(
            "Browser",
            options=st.session_state.data['browser'].unique()
        )
    with col3:
        clicked_filter = st.selectbox(
            "Clicked",
            options=['All', 'Yes', 'No']
        )
    
    # Apply filters
    filtered_data = st.session_state.data.copy()
    if device_filter:
        filtered_data = filtered_data[filtered_data['device_type'].isin(device_filter)]
    if browser_filter:
        filtered_data = filtered_data[filtered_data['browser'].isin(browser_filter)]
    if clicked_filter != 'All':
        filtered_data = filtered_data[filtered_data['clicked'] == (clicked_filter == 'Yes')]
    
    # Display data
    st.dataframe(filtered_data, use_container_width=True)
    
    # Data statistics
    st.subheader("Data Statistics")
    st.write(filtered_data.describe())
