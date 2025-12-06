"""
Monitoring Dashboard Module
Provides Streamlit components for visualizing metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .metrics import MetricsCollector


class MonitoringDashboard:
    """
    Streamlit dashboard for monitoring metrics
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize monitoring dashboard
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics = metrics_collector
    
    def render_metrics_overview(self):
        """Render metrics overview section"""
        st.markdown("## 📊 Metrics Overview")
        
        summary = self.metrics.get_metrics_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Predictions",
                f"{summary['total_predictions']:,}",
                help="Total number of predictions made"
            )
        
        with col2:
            st.metric(
                "Success Rate",
                f"{summary['success_rate_percent']:.1f}%",
                delta=f"{summary['successful_predictions']} successful",
                delta_color="normal",
                help="Percentage of successful predictions"
            )
        
        with col3:
            st.metric(
                "Avg Latency",
                f"{summary['latency']['mean_ms']:.1f} ms",
                delta=f"P95: {summary['latency']['p95_ms']:.1f} ms",
                help="Average inference latency"
            )
        
        with col4:
            st.metric(
                "Throughput",
                f"{summary['throughput_per_minute']:.1f}/min",
                help="Predictions per minute"
            )
    
    def render_inference_metrics(self):
        """Render inference metrics section"""
        st.markdown("## 🎯 Inference Metrics")
        
        inference = self.metrics.get_inference_metrics()
        latency = inference['inference_latency']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Latency Statistics")
            
            # Latency metrics table
            latency_df = pd.DataFrame([
                {"Metric": "Mean", "Value (ms)": latency['mean_ms']},
                {"Metric": "Median", "Value (ms)": latency['median_ms']},
                {"Metric": "P95", "Value (ms)": latency['p95_ms']},
                {"Metric": "P99", "Value (ms)": latency['p99_ms']},
                {"Metric": "Min", "Value (ms)": latency['min_ms']},
                {"Metric": "Max", "Value (ms)": latency['max_ms']},
            ])
            st.dataframe(latency_df, hide_index=True, use_container_width=True)
            
            # Latency distribution chart
            if self.metrics.latencies:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=list(self.metrics.latencies),
                    nbinsx=30,
                    name="Latency Distribution",
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title="Latency Distribution",
                    xaxis_title="Latency (ms)",
                    yaxis_title="Frequency",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Prediction Distribution")
            
            distribution = inference['prediction_distribution']
            dist_percent = inference['distribution_percentage']
            
            if distribution:
                # Pie chart
                labels = ['No Diabetes', 'Diabetes']
                values = [distribution.get(0, 0), distribution.get(1, 0)]
                colors = ['#51cf66', '#ff6b6b']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Prediction Class Distribution",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution table
                dist_df = pd.DataFrame([
                    {"Class": "No Diabetes (0)", "Count": distribution.get(0, 0), "Percentage": f"{dist_percent.get(0, 0):.1f}%"},
                    {"Class": "Diabetes (1)", "Count": distribution.get(1, 0), "Percentage": f"{dist_percent.get(1, 0):.1f}%"},
                ])
                st.dataframe(dist_df, hide_index=True, use_container_width=True)
    
    def render_error_metrics(self):
        """Render error metrics section"""
        st.markdown("## ⚠️ Error Metrics")
        
        error_metrics = self.metrics.get_error_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Errors",
                error_metrics['total_errors'],
                help="Total number of failed predictions"
            )
        
        with col2:
            st.metric(
                "Error Rate",
                f"{error_metrics['error_rate_percent']:.2f}%",
                delta_color="inverse",
                help="Percentage of failed predictions"
            )
        
        with col3:
            st.metric(
                "Errors (1h)",
                error_metrics['recent_errors_1h'],
                help="Errors in the last hour"
            )
        
        # Error breakdown
        if error_metrics['error_breakdown']:
            st.markdown("### Error Breakdown by Type")
            
            error_df = pd.DataFrame([
                {"Error Type": k, "Count": v}
                for k, v in error_metrics['error_breakdown'].items()
            ])
            
            fig = px.bar(
                error_df,
                x='Error Type',
                y='Count',
                color='Error Type',
                title="Error Distribution by Type"
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent errors
        if error_metrics['latest_errors']:
            st.markdown("### Recent Errors")
            errors_df = pd.DataFrame(error_metrics['latest_errors'])
            st.dataframe(errors_df, use_container_width=True, hide_index=True)
        else:
            st.success("✓ No recent errors")
    
    def render_throughput_metrics(self):
        """Render throughput metrics section"""
        st.markdown("## 🚀 Throughput Metrics")
        
        throughput = self.metrics.get_throughput_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Last 1 min",
                f"{throughput['throughput_1min']}",
                help="Predictions in last minute"
            )
        
        with col2:
            st.metric(
                "Last 5 min",
                f"{throughput['throughput_5min']:.1f}/min",
                help="Average predictions per minute (last 5 min)"
            )
        
        with col3:
            st.metric(
                "Last 1 hour",
                f"{throughput['throughput_1hour']:.1f}/min",
                help="Average predictions per minute (last hour)"
            )
        
        with col4:
            st.metric(
                "Total Requests",
                f"{throughput['total_requests']:,}",
                help="Total requests processed"
            )
    
    def render_recent_predictions(self, limit: int = 50):
        """Render recent predictions table"""
        st.markdown("## 📝 Recent Predictions")
        
        predictions = self.metrics.get_recent_predictions(limit)
        
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Format columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Select columns to display
            display_cols = [
                'timestamp', 'prediction_id', 'prediction_value', 
                'probability', 'confidence', 'latency_ms', 'success'
            ]
            display_cols = [col for col in display_cols if col in df.columns]
            
            st.dataframe(
                df[display_cols].tail(limit),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No predictions recorded yet")
    
    def render_latency_timeline(self):
        """Render latency over time chart"""
        st.markdown("## ⏱️ Latency Timeline")
        
        predictions = self.metrics.get_recent_predictions(200)
        
        if predictions:
            df = pd.DataFrame(predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            # Successful predictions
            success_df = df[df['success'] == True]
            if not success_df.empty:
                fig.add_trace(go.Scatter(
                    x=success_df['timestamp'],
                    y=success_df['latency_ms'],
                    mode='lines+markers',
                    name='Successful',
                    line=dict(color='#51cf66', width=2),
                    marker=dict(size=6)
                ))
            
            # Failed predictions
            failed_df = df[df['success'] == False]
            if not failed_df.empty:
                fig.add_trace(go.Scatter(
                    x=failed_df['timestamp'],
                    y=failed_df['latency_ms'],
                    mode='markers',
                    name='Failed',
                    marker=dict(color='#ff6b6b', size=8, symbol='x')
                ))
            
            fig.update_layout(
                title="Inference Latency Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Latency (ms)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available for timeline")
    
    def render_full_dashboard(self):
        """Render complete monitoring dashboard"""
        st.title("📊 ML Model Monitoring Dashboard")
        st.markdown("Real-time monitoring of diabetes prediction model")
        
        # Auto-refresh option
        col1, col2 = st.columns([3, 1])
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.rerun()
        
        st.divider()
        
        # Overview
        self.render_metrics_overview()
        st.divider()
        
        # Two columns layout
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_inference_metrics()
        
        with col2:
            self.render_throughput_metrics()
            st.markdown("---")
            self.render_error_metrics()
        
        st.divider()
        
        # Latency timeline
        self.render_latency_timeline()
        
        st.divider()
        
        # Recent predictions
        self.render_recent_predictions()
        
        # Export metrics
        st.divider()
        st.markdown("### 💾 Export Metrics")
        if st.button("Export Metrics to JSON"):
            filepath = self.metrics.export_metrics()
            st.success(f"✓ Metrics exported to: {filepath}")
            
            # Offer download
            with open(filepath, 'r') as f:
                st.download_button(
                    label="📥 Download Metrics JSON",
                    data=f.read(),
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
