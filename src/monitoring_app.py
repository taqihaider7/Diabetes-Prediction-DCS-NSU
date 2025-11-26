"""
Separate Monitoring Dashboard Application
Real-time monitoring for Diabetes Prediction ML Model

This is a standalone Streamlit app dedicated to monitoring and metrics visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring import MetricsCollector, MonitoringDashboard, get_logger
from monitoring.config import DEFAULT_CONFIG
from monitoring.shared_metrics import get_shared_store

# Configure page
st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
    }
    .error-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    }
    .warning-card {
        background: linear-gradient(135deg, #ffd93d 0%, #f6c23e 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_logger(__name__)

# Load or create metrics collector
@st.cache_resource
def get_metrics_collector():
    """Get or create metrics collector instance"""
    return MetricsCollector(retention_hours=DEFAULT_CONFIG.metrics_retention_hours)

metrics_collector = get_metrics_collector()

# Sidebar
with st.sidebar:
    st.title("⚙️ Monitoring Settings")
    
    # Navigation
    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🎯 Inference Metrics",
            "⚠️ Error Analysis",
            "🚀 Performance",
            "📝 Recent Activity",
            "📈 Analytics",
            "⚙️ Configuration"
        ]
    )
    
    st.divider()
    
    # Quick Stats - Read from shared storage
    st.markdown("### 📈 Quick Stats")
    shared_store = get_shared_store()
    summary = shared_store.get_summary()
    
    st.metric("Total Predictions", f"{summary['total_predictions']:,}")
    st.metric("Success Rate", f"{summary['success_rate_percent']:.1f}%")
    st.metric("Avg Latency", f"{summary['latency']['mean_ms']:.1f} ms")
    
    st.divider()
    
    # Refresh controls
    st.markdown("### 🔄 Refresh")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    if st.button("🔄 Refresh Now", width='stretch'):
        st.rerun()
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    st.divider()
    
    # Export options
    st.markdown("### 💾 Export")
    if st.button("Export Metrics", width='stretch'):
        filepath = metrics_collector.export_metrics()
        st.success(f"✓ Exported to: {os.path.basename(filepath)}")
        
        with open(filepath, 'r') as f:
            st.download_button(
                "📥 Download JSON",
                data=f.read(),
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width=True
            )
    
    # Reset metrics
    if st.button("🗑️ Reset Metrics", width='stretch'):
        metrics_collector.reset_metrics()
        st.success("✓ Metrics reset")
        st.rerun()
    
    st.divider()
    
    # Connection status
    st.markdown("### 🔌 Connection Status")
    
    # Grafana Cloud status
    grafana_status = "✅ Connected" if DEFAULT_CONFIG.loki_enabled else "❌ Disabled"
    st.caption(f"Grafana Cloud: {grafana_status}")
    
    # Environment
    st.caption(f"Environment: {DEFAULT_CONFIG.environment}")
    st.caption(f"App: {DEFAULT_CONFIG.app_name}")

# Main content
st.title("📊 ML Model Monitoring Dashboard")
st.markdown(f"**Real-time monitoring** for Diabetes Prediction System | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== PAGE: Overview ====================
if page == "📊 Overview":
    st.divider()
    
    # Create dashboard instance with shared data
    shared_store = get_shared_store()
    summary = shared_store.get_summary()
    
    # Metrics Overview
    st.markdown("### 📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{summary['total_predictions']:,}")
    
    with col2:
        st.metric("Successful", f"{summary['successful_predictions']:,}", 
                 delta=f"{summary['success_rate_percent']:.1f}%")
    
    with col3:
        st.metric("Failed", f"{summary['failed_predictions']:,}",
                 delta=f"-{100-summary['success_rate_percent']:.1f}%" if summary['total_predictions'] > 0 else "0%")
    
    with col4:
        st.metric("Avg Latency", f"{summary['latency']['mean_ms']:.1f} ms")
    
    st.divider()
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Prediction Distribution")
        distribution = summary['prediction_distribution']
        
        if distribution and sum(distribution.values()) > 0:
            labels = ['No Diabetes (0)', 'Diabetes (1)']
            values = [distribution.get(0, 0), distribution.get(1, 0)]
            colors = ['#51cf66', '#ff6b6b']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent+value',
                textposition='outside'
            )])
            fig.update_layout(
                height=350,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("📊 No predictions yet. Start making predictions to see distribution.")
    
    with col2:
        st.markdown("### ⏱️ Latency Statistics")
        latency = summary['latency']
        
        latency_df = pd.DataFrame([
            {"Metric": "Mean", "Value (ms)": latency['mean_ms'], "Category": "Average"},
            {"Metric": "Median", "Value (ms)": latency['median_ms'], "Category": "Average"},
            {"Metric": "P95", "Value (ms)": latency['p95_ms'], "Category": "Percentile"},
            {"Metric": "P99", "Value (ms)": latency['p99_ms'], "Category": "Percentile"},
            {"Metric": "Min", "Value (ms)": latency['min_ms'], "Category": "Extremes"},
            {"Metric": "Max", "Value (ms)": latency['max_ms'], "Category": "Extremes"},
        ])
        
        fig = px.bar(
            latency_df,
            x='Metric',
            y='Value (ms)',
            color='Category',
            title='Latency Breakdown',
            text='Value (ms)',
            color_discrete_map={
                'Average': '#667eea',
                'Percentile': '#764ba2',
                'Extremes': '#f6c23e'
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Latency timeline
    shared_store = get_shared_store()
    predictions = shared_store.get_all_predictions(limit=100)
    
    if predictions:
        st.markdown("### ⏱️ Latency Timeline")
        
        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['latency_ms'],
            mode='lines+markers',
            name='Latency',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Prediction Latency Over Time',
            xaxis_title='Time',
            yaxis_title='Latency (ms)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("📊 No predictions yet. Start making predictions to see timeline.")

# ==================== PAGE: Inference Metrics ====================
elif page == "🎯 Inference Metrics":
    st.divider()
    
    shared_store = get_shared_store()
    summary = shared_store.get_summary()
    
    st.markdown("## 🎯 Inference Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Inferences", f"{summary['total_predictions']:,}")
    
    with col2:
        success_rate = summary['success_rate_percent']
        st.metric("Success Rate", f"{success_rate:.1f}%",
                 delta="Good" if success_rate > 95 else "Check errors")
    
    with col3:
        st.metric("Total Errors", f"{summary['total_errors']:,}")
    
    st.divider()
    
    # Prediction distribution
    distribution = summary['prediction_distribution']
    
    if distribution and sum(distribution.values()) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Distribution")
            labels = ['No Diabetes (0)', 'Diabetes (1)']
            values = [distribution.get(0, 0), distribution.get(1, 0)]
            
            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition='auto',
                marker_color=['#51cf66', '#ff6b6b']
            )])
            fig.update_layout(
                yaxis_title='Count',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### 📈 Distribution Percentage")
            total = sum(values)
            if total > 0:
                percentages = [v/total*100 for v in values]
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=['#51cf66', '#ff6b6b']),
                    hole=0.4
                )])
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')
    else:
        st.info("📊 No predictions yet.")

# ==================== PAGE: Error Analysis ====================
elif page == "⚠️ Error Analysis":
    st.divider()
    dashboard = MonitoringDashboard(metrics_collector)
    dashboard.render_error_metrics()
    
    st.divider()
    
    # Additional error insights
    error_metrics = metrics_collector.get_error_metrics()
    
    if error_metrics['total_errors'] > 0:
        st.markdown("### 🔍 Error Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error trend
            st.markdown("#### Error Rate Trend")
            recent_errors = error_metrics['recent_errors_1h']
            total_predictions = summary['total_predictions']
            
            if total_predictions > 0:
                error_rate_1h = (recent_errors / total_predictions * 100) if total_predictions > 0 else 0
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=error_rate_1h,
                    title="Error Rate (Last Hour) %",
                    delta={'reference': 5, 'relative': False},
                    gauge={
                        'axis': {'range': [0, 20]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgreen"},
                            {'range': [2, 5], 'color': "lightyellow"},
                            {'range': [5, 20], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 5
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Error types pie chart
            if error_metrics['error_breakdown']:
                st.markdown("#### Error Distribution")
                
                error_types = list(error_metrics['error_breakdown'].keys())
                error_counts = list(error_metrics['error_breakdown'].values())
                
                fig = go.Figure(data=[go.Pie(
                    labels=error_types,
                    values=error_counts,
                    hole=0.3,
                    textinfo='label+percent'
                )])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, width='stretch')

# ==================== PAGE: Performance ====================
elif page == "🚀 Performance":
    st.divider()
    dashboard = MonitoringDashboard(metrics_collector)
    dashboard.render_throughput_metrics()
    
    st.divider()
    
    # Performance analysis
    st.markdown("### 📈 Performance Analysis")
    
    throughput = metrics_collector.get_throughput_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Throughput trend
        st.markdown("#### Throughput Trend")
        
        time_windows = ['1 min', '5 min', '1 hour']
        throughput_values = [
            throughput['throughput_1min'],
            throughput['throughput_5min'],
            throughput['throughput_1hour']
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=time_windows,
                y=throughput_values,
                text=throughput_values,
                texttemplate='%{text:.1f}',
                textposition='outside',
                marker_color=['#51cf66', '#667eea', '#764ba2']
            )
        ])
        fig.update_layout(
            title='Predictions per Minute',
            yaxis_title='Predictions/min',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Latency distribution
        st.markdown("#### Latency Distribution")
        
        if metrics_collector.latencies:
            fig = go.Figure(data=[go.Histogram(
                x=list(metrics_collector.latencies),
                nbinsx=30,
                marker_color='#667eea',
                opacity=0.75
            )])
            fig.update_layout(
                title='Latency Distribution',
                xaxis_title='Latency (ms)',
                yaxis_title='Frequency',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No latency data available yet")

# ==================== PAGE: Recent Activity ====================
elif page == "📝 Recent Activity":
    st.divider()
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        limit = st.selectbox("Show records", [10, 25, 50, 100], index=2)
    
    with col2:
        filter_type = st.selectbox("Filter by", ["All", "Successful", "Failed"])
    
    with col3:
        if st.button("🔄 Refresh", width='stretch'):
            st.rerun()
    
    # Get predictions from shared storage
    shared_store = get_shared_store()
    predictions = shared_store.get_all_predictions(limit=limit)
    
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Apply filter
        if filter_type == "Successful":
            df = df[df['success'] == True]
        elif filter_type == "Failed":
            df = df[df['success'] == False]
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add status emoji
        df['status_icon'] = df['success'].apply(lambda x: '✅' if x else '❌')
        
        # Select and reorder columns
        display_cols = ['status_icon', 'timestamp', 'prediction_id', 'prediction_value', 
                       'probability', 'confidence', 'latency_ms', 'user_session']
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Rename columns for display
        df_display = df[display_cols].copy()
        df_display.columns = ['Status', 'Time', 'ID', 'Prediction', 'Probability', 
                              'Confidence', 'Latency (ms)', 'Session'][:len(display_cols)]
        
        st.dataframe(
            df_display,
            width=True,
            hide_index=True,
            height=600
        )
        
        # Summary stats for filtered data
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Records", len(df))
        with col2:
            avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else 0
            st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        with col3:
            success_count = df['success'].sum() if 'success' in df.columns else 0
            st.metric("Successful", success_count)
        with col4:
            failed_count = len(df) - success_count
            st.metric("Failed", failed_count)
    else:
        st.info("📊 No predictions recorded yet. Make some predictions to see activity.")

# ==================== PAGE: Analytics ====================
elif page == "📈 Analytics":
    st.divider()
    st.markdown("## 📈 Advanced Analytics")
    
    shared_store = get_shared_store()
    predictions = shared_store.get_all_predictions(limit=500)
    
    if predictions and len(predictions) > 0:
        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based analysis
        st.markdown("### ⏰ Time-Based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Predictions over time
            df_time = df.set_index('timestamp')
            df_hourly = df_time.resample('5T').size().reset_index()
            df_hourly.columns = ['Time', 'Count']
            
            fig = px.line(
                df_hourly,
                x='Time',
                y='Count',
                title='Predictions Over Time (5-min intervals)',
                markers=True
            )
            fig.update_traces(line_color='#667eea', line_width=2)
            fig.update_layout(height=350)
            st.plotly_chart(fig, width=True)
        
        with col2:
            # Success rate over time
            df_success = df_time.resample('5T')['success'].agg(['sum', 'count']).reset_index()
            df_success['success_rate'] = (df_success['sum'] / df_success['count'] * 100).fillna(0)
            
            fig = px.line(
                df_success,
                x='timestamp',
                y='success_rate',
                title='Success Rate Over Time (%)',
                markers=True
            )
            fig.update_traces(line_color='#51cf66', line_width=2)
            fig.update_yaxis(range=[0, 100])
            fig.update_layout(height=350)
            st.plotly_chart(fig, width=True)
        
        st.divider()
        
        # Statistical analysis
        st.markdown("### 📊 Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Probability distribution
            st.markdown("#### Probability Distribution")
            successful_df = df[df['success'] == True]
            
            if len(successful_df) > 0:
                fig = go.Figure(data=[go.Histogram(
                    x=successful_df['probability'],
                    nbinsx=20,
                    marker_color='#764ba2'
                )])
                fig.update_layout(
                    xaxis_title='Probability',
                    yaxis_title='Frequency',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, width=True)
        
        with col2:
            # Confidence distribution
            st.markdown("#### Confidence Distribution")
            
            if len(successful_df) > 0:
                fig = go.Figure(data=[go.Histogram(
                    x=successful_df['confidence'],
                    nbinsx=20,
                    marker_color='#667eea'
                )])
                fig.update_layout(
                    xaxis_title='Confidence',
                    yaxis_title='Frequency',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, width=True)
        
        with col3:
            # Latency box plot
            st.markdown("#### Latency by Result")
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[df['success'] == True]['latency_ms'],
                name='Success',
                marker_color='#51cf66'
            ))
            fig.add_trace(go.Box(
                y=df[df['success'] == False]['latency_ms'],
                name='Failed',
                marker_color='#ff6b6b'
            ))
            fig.update_layout(
                yaxis_title='Latency (ms)',
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig, width=True)
        
        st.divider()
        
        # Correlation analysis
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_cols = ['prediction_value', 'probability', 'confidence', 'latency_ms']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(numeric_cols) > 1:
            corr_matrix = df[df['success'] == True][numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix (Successful Predictions)'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width=True)
    else:
        st.info("📊 Not enough data for analytics. Make more predictions to see insights.")

# ==================== PAGE: Configuration ====================
elif page == "⚙️ Configuration":
    st.divider()
    st.markdown("## ⚙️ Configuration")
    
    # Current configuration
    st.markdown("### 📋 Current Settings")
    
    config_data = {
        "Setting": [
            "Application Name",
            "Environment",
            "Loki Enabled",
            "Loki URL",
            "Log Level",
            "Metrics Retention",
            "User Session Tracking"
        ],
        "Value": [
            DEFAULT_CONFIG.app_name,
            DEFAULT_CONFIG.environment,
            "✅ Yes" if DEFAULT_CONFIG.loki_enabled else "❌ No",
            DEFAULT_CONFIG.loki_url if DEFAULT_CONFIG.loki_enabled else "N/A",
            DEFAULT_CONFIG.log_level,
            f"{DEFAULT_CONFIG.metrics_retention_hours} hours",
            "✅ Yes" if DEFAULT_CONFIG.track_user_sessions else "❌ No"
        ]
    }
    
    st.dataframe(pd.DataFrame(config_data), width=True, hide_index=True)
    
    st.divider()
    
    # Grafana Cloud setup
    st.markdown("### ☁️ Grafana Cloud Setup")
    
    st.markdown("""
    **How to connect to Grafana Cloud:**
    
    1. **Sign up for Grafana Cloud** (free tier available)
       - Visit: https://grafana.com/auth/sign-up/create-user
    
    2. **Get your Loki endpoint**
       - Navigate to: Connections → Data Sources → Loki
       - Copy the URL (format: `https://logs-prod-xxx.grafana.net/loki/api/v1/push`)
    
    3. **Create API token**
       - Go to: Cloud Portal → API Keys
       - Create a new API key with `MetricsPublisher` role
    
    4. **Configure environment variables**
       ```bash
       LOKI_ENABLED=true
       LOKI_URL=https://YOUR-ID.grafana.net/loki/api/v1/push
       LOKI_USERNAME=your-instance-id
       LOKI_API_KEY=your-api-key
       ```
    
    5. **Update logger configuration** in `src/monitoring/logger.py`
    """)
    
    # Test connection button
    if st.button("🔌 Test Grafana Cloud Connection", width='stretch'):
        if DEFAULT_CONFIG.loki_enabled:
            try:
                import requests
                response = requests.get(
                    DEFAULT_CONFIG.loki_url.replace('/loki/api/v1/push', '/ready'),
                    timeout=5
                )
                if response.status_code == 200:
                    st.success("✅ Successfully connected to Grafana Cloud!")
                else:
                    st.error(f"❌ Connection failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
        else:
            st.warning("⚠️ Loki is disabled in configuration")
    
    st.divider()
    
    # System information
    st.markdown("### 💻 System Information")
    
    import platform
    
    system_info = {
        "Property": [
            "Python Version",
            "Platform",
            "Streamlit Version",
            "Current Time",
            "Uptime"
        ],
        "Value": [
            platform.python_version(),
            platform.system(),
            st.__version__,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "N/A"  # Would need app start time tracking
        ]
    }
    
    st.dataframe(pd.DataFrame(system_info), width='stretch', hide_index=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>ML Monitoring Dashboard v1.0</strong> | Diabetes Prediction System</p>
        <p>Built with Streamlit, Grafana Cloud, and ❤️</p>
    </div>
""", unsafe_allow_html=True)
