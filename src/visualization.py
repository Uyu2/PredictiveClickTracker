import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class DashboardVisualizer:
    @staticmethod
    def create_feature_importance_plot(feature_importance_df):
        fig = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Key Factors Influencing User Clicks',
            labels={
                'importance': 'Impact Score',
                'feature': 'Factor'
            },
            color='importance',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=20,
            showlegend=False,
            height=400,
            margin=dict(t=50, l=0, r=0, b=0),
            yaxis_title="",
            xaxis_title="Impact on Click Probability"
        )

        return fig

    @staticmethod
    def create_search_frequency_plot(df):
        search_counts = df['search_term'].value_counts().reset_index()
        search_counts.columns = ['Product', 'Search Count']

        # Calculate click-through rate for each product
        product_ctr = df.groupby('search_term')['clicked'].mean().reset_index()
        product_ctr.columns = ['Product', 'Click Rate']

        # Merge search counts with click-through rates and limit to top 15
        search_data = search_counts.merge(product_ctr, on='Product')
        search_data['Click Rate'] = search_data['Click Rate'] * 100
        search_data = search_data.head(15)  # Limit to top 15 products

        fig = px.bar(
            search_data,
            x='Product',
            y='Search Count',
            title='Top 15 Most Searched Products and Their Success Rates',
            color='Click Rate',
            labels={
                'Product': 'Product Name',
                'Search Count': 'Number of Searches',
                'Click Rate': 'Click-through Rate (%)'
            },
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=18,
            height=500,
            margin=dict(t=50, l=0, r=0, b=0),
            xaxis_tickangle=45,
            showlegend=True
        )

        return fig

    @staticmethod
    def create_click_rate_by_category(df, category):
        click_rates = df.groupby(category)['clicked'].agg(['mean', 'count']).reset_index()
        click_rates.columns = [category, 'click_rate', 'sample_size']
        click_rates['click_rate_pct'] = click_rates['click_rate'] * 100

        fig = px.bar(
            click_rates,
            x=category,
            y='click_rate',
            title=f'Click-through Rate by {category.replace("_", " ").title()}',
            labels={
                'click_rate': 'Click-through Rate (%)',
                category: category.replace('_', ' ').title()
            },
            text=click_rates['click_rate_pct'].round(1).astype(str) + '%',
            color='click_rate',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            title_x=0.5,
            title_font_size=18,
            showlegend=False,
            height=400,
            margin=dict(t=50, l=0, r=0, b=0)
        )

        fig.update_traces(textposition='outside')

        return fig

    @staticmethod
    def create_time_series_plot(df):
        daily_clicks = df.groupby(df['timestamp'].dt.date).agg({
            'clicked': ['mean', 'count']
        }).reset_index()
        daily_clicks.columns = ['date', 'click_rate', 'sample_size']
        daily_clicks['click_rate_pct'] = daily_clicks['click_rate'] * 100

        fig = go.Figure()

        # Add the line for click rate
        fig.add_trace(go.Scatter(
            x=daily_clicks['date'],
            y=daily_clicks['click_rate'],
            mode='lines+markers',
            name='Click Rate',
            line=dict(color='#2E86C1', width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title='Daily Click-through Rate Trend',
            title_x=0.5,
            title_font_size=18,
            height=400,
            margin=dict(t=50, l=0, r=0, b=0),
            xaxis_title="Date",
            yaxis_title="Click-through Rate",
            yaxis_tickformat='.1%',
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_correlation_heatmap(df):
        numeric_cols = ['time_on_screen', 'exited_screen', 'search_count', 'clicked']
        corr_matrix = df[numeric_cols].corr()

        # Make the column names more readable
        readable_names = {
            'time_on_screen': 'Time on Screen',
            'exited_screen': 'Bounced',
            'search_count': 'Searches Made',
            'clicked': 'Clicked'
        }

        corr_matrix.columns = [readable_names[col] for col in corr_matrix.columns]
        corr_matrix.index = [readable_names[col] for col in corr_matrix.index]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Relationship Between Different Metrics',
            title_x=0.5,
            title_font_size=18,
            height=400,
            margin=dict(t=50, l=0, r=0, b=0)
        )

        return fig
    
    @staticmethod
    def create_decision_tree_visualization(model, data):
        # Create a simplified decision tree visualization
        node_trace = go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 1.5, 1.8, 1.2],
            mode='markers+text',
            marker=dict(size=30, color='lightblue'),
            text=['User Visit', 'Desktop?', 'Long Visit?', 'Many Searches?', 'Clicked!'],
            textposition='bottom center'
        )

        # Add connecting lines
        edge_trace = go.Scatter(
            x=[1, 2, 2, 3, 3, 4, 4, 5],
            y=[1, 2, 2, 1.5, 1.5, 1.8, 1.8, 1.2],
            mode='lines',
            line=dict(width=2, color='gray'),
            hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Simplified Decision Path Visualization',
            title_x=0.5,
            showlegend=False,
            height=400,
            margin=dict(t=50, l=0, r=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )

        return fig