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

# Branding
st.markdown("""
    <div style='position: fixed; top: 0; left: 0; padding: 10px; color: #666; font-size: 20px; font-weight: 300;'>
    Wyndo Search
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    data_gen = DataGenerator(n_samples=1000)
    st.session_state.data = data_gen.generate_data()
    st.session_state.model = ClickPredictionModel()
    st.session_state.model_score = st.session_state.model.train(st.session_state.data)

# Sidebar
st.sidebar.title("Dashboard Controls")
sample_size = st.sidebar.slider("Sample Size", 100, 2000, 1000, 100)
if st.sidebar.button("Regenerate Data"):
    data_gen = DataGenerator(n_samples=sample_size)
    st.session_state.data = data_gen.generate_data()
    st.session_state.model_score = st.session_state.model.train(st.session_state.data)

# Main content
tab1, tab2, tab3 = st.tabs(["Analytics Dashboard", "Model Insights", "Dataset Explorer"])

with tab1:
    st.title("Click-through Rate Analytics Dashboard")

    # Key Terms and Definitions
    with st.expander("Key Terms and Definitions"):
        st.markdown("""
        **Important Metrics Explained:**
        - **Click-through Rate (CTR)**: The percentage of users who click on a product after viewing it
        - **Bounce Rate**: Percentage of users who leave the page without any interaction. When a user bounces (exits), 
          they cannot have clicked on a product, as they left before taking any action
        - **Time on Screen**: How long users spend viewing a product before making a decision
        - **Search Count**: Number of different products a user looked for

        **Device Types:**
        - **Desktop**: Traditional computer access
        - **Mobile**: Smartphone access
        - **Tablet**: iPad or similar device access

        **Traffic Sources:**
        - **Direct**: Users coming directly to the site
        - **Search**: Users from search engines
        - **Social**: Users from social media
        - **Email**: Users from email campaigns
        """)

    st.markdown("""
    This dashboard analyzes user behavior and predicts click-through rates using advanced machine learning techniques.
    Our predictive model achieves {:.1%} accuracy in determining which factors lead to successful product interactions.
    """.format(st.session_state.model_score))

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_ctr = st.session_state.data['clicked'].mean()
        st.metric("Average Click Rate", f"{avg_ctr:.2%}")
    with col2:
        avg_time = st.session_state.data['time_on_screen'].mean()
        st.metric("Avg Time on Screen", f"{avg_time:.1f}s")
    with col3:
        bounce_rate = st.session_state.data['exited_screen'].mean()
        st.metric("Bounce Rate", f"{bounce_rate:.2%}")

    # Feature importance analysis
    st.subheader("What Drives User Clicks?")
    st.markdown("""
    Using advanced machine learning (Random Forest), we've identified the key factors that influence 
    whether a user will click on a product. This analysis helps us understand which aspects of the 
    user experience have the biggest impact on customer engagement.
    """)
    feature_importance_plot = DashboardVisualizer.create_feature_importance_plot(
        st.session_state.model.feature_importance
    )
    st.plotly_chart(feature_importance_plot, use_container_width=True)

    # Product search analysis
    st.subheader("Most Searched Products")
    st.markdown("""
    Understanding which products attract the most customer interest helps optimize 
    inventory and marketing strategies.
    """)
    search_frequency_plot = DashboardVisualizer.create_search_frequency_plot(
        st.session_state.data
    )
    st.plotly_chart(search_frequency_plot, use_container_width=True)

    # Click-through rate by category
    st.subheader("User Behavior Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Click Rates by Device")
        st.markdown("Compare how users on different devices interact with products")
        device_plot = DashboardVisualizer.create_click_rate_by_category(
            st.session_state.data, 'device_type'
        )
        st.plotly_chart(device_plot, use_container_width=True)

    with col2:
        st.markdown("#### Browser Performance")
        st.markdown("Analysis of click-through rates across different web browsers")
        browser_plot = DashboardVisualizer.create_click_rate_by_category(
            st.session_state.data, 'browser'
        )
        st.plotly_chart(browser_plot, use_container_width=True)

    # Time series
    st.subheader("Click Rate Trends Over Time")
    st.markdown("Track how customer engagement changes over the analysis period")
    time_series = DashboardVisualizer.create_time_series_plot(st.session_state.data)
    st.plotly_chart(time_series, use_container_width=True)

with tab2:
    st.title("Model Insights & Predictions")

    st.markdown("""
    ### Understanding Our Predictive Models

    We use two powerful machine learning models to analyze and predict customer behavior:

    1. **Random Forest Model**
    - Combines multiple decision trees for robust predictions
    - Analyzes patterns across different customer behaviors
    - Provides reliable importance scores for different factors

    2. **Decision Tree Analysis**
    - Creates clear decision pathways
    - Helps understand specific customer segments
    - Easier to interpret and explain
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Performance")
        st.markdown(f"""
        Our current model achieves:
        - **{st.session_state.model_score:.1%}** prediction accuracy
        - Trained on {len(st.session_state.data):,} customer interactions
        - Updated with latest data
        """)

    with col2:
        st.markdown("### Key Findings")
        st.markdown("""
        - Desktop users show highest engagement
        - Time spent on product pages strongly indicates interest
        - Browser choice significantly impacts click rates
        """)

    # Decision paths visualization
    st.subheader("Customer Decision Pathways")
    st.markdown("""
    This visualization shows how different factors combine to influence customer decisions.
    Thicker lines indicate more common customer journeys.
    """)
    decision_tree_viz = DashboardVisualizer.create_decision_tree_visualization(
        st.session_state.model, st.session_state.data
    )
    st.plotly_chart(decision_tree_viz, use_container_width=True)

with tab3:
    st.title("Dataset Explorer")

    # Filters
    st.markdown("### Filter the Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        device_filter = st.multiselect(
            "Device Type",
            options=sorted(st.session_state.data['device_type'].unique())
        )
    with col2:
        browser_filter = st.multiselect(
            "Browser",
            options=sorted(st.session_state.data['browser'].unique())
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

    # Convert timestamp to string for display
    display_data = filtered_data.copy()
    display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Display data
    st.markdown("### Data Records")
    st.dataframe(display_data, use_container_width=True)

    # Data statistics
    st.subheader("Statistical Summary")
    st.write(filtered_data.describe())