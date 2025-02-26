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
            title='Feature Importance in Click-through Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        return fig
    
    @staticmethod
    def create_click_rate_by_category(df, category):
        click_rates = df.groupby(category)['clicked'].mean().reset_index()
        fig = px.bar(
            click_rates,
            x=category,
            y='clicked',
            title=f'Click-through Rate by {category.title()}',
            labels={'clicked': 'Click-through Rate'}
        )
        return fig
    
    @staticmethod
    def create_time_series_plot(df):
        daily_clicks = df.groupby(df['timestamp'].dt.date)['clicked'].mean().reset_index()
        fig = px.line(
            daily_clicks,
            x='timestamp',
            y='clicked',
            title='Click-through Rate Over Time',
            labels={'clicked': 'Click-through Rate', 'timestamp': 'Date'}
        )
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df):
        numeric_cols = ['time_on_screen', 'exited_screen', 'search_count', 'clicked']
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            width=600,
            height=600
        )
        
        return fig
