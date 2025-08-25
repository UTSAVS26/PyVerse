#!/usr/bin/env python3
"""
HoneypotAI Dashboard
Real-time monitoring and visualization dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeypot import HoneypotManager
from ml import ThreatDetector
from adapt import AdaptiveResponse

# Page configuration
st.set_page_config(
    page_title="HoneypotAI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class HoneypotAIDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.honeypot_manager = HoneypotManager()
        self.threat_detector = ThreatDetector()
        self.adaptive_response = AdaptiveResponse()
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = 30
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🛡️ HoneypotAI Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced Adaptive Cybersecurity Intelligence Platform")
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🎛️ Controls")
        
        # Refresh rate
        refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)",
            min_value=5,
            max_value=60,
            value=st.session_state.refresh_rate,
            step=5
        )
        st.session_state.refresh_rate = refresh_rate
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # System controls
        st.sidebar.subheader("⚙️ System Controls")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start System"):
                st.success("System started!")
        
        with col2:
            if st.button("⏹️ Stop System"):
                st.error("System stopped!")
        
        # Export options
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Export Data")
        
        if st.sidebar.button("📥 Export Logs"):
            st.info("Export functionality would be implemented here")
        
        if st.sidebar.button("📈 Export Reports"):
            st.info("Report export would be implemented here")
    
    def render_overview_metrics(self):
        """Render overview metrics"""
        st.subheader("📊 System Overview")
        
        # Get system status
        honeypot_stats = self.honeypot_manager.get_overall_stats()
        ml_stats = self.threat_detector.get_detection_stats()
        response_stats = self.adaptive_response.get_status()
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Connections",
                value=honeypot_stats.get('total_connections', 0),
                delta=honeypot_stats.get('total_connections', 0) - 0
            )
        
        with col2:
            st.metric(
                label="Threat Detections",
                value=ml_stats.get('total_detections', 0),
                delta=ml_stats.get('total_detections', 0) - 0
            )
        
        with col3:
            st.metric(
                label="Blocked IPs",
                value=response_stats.get('blocked_ips_count', 0),
                delta=response_stats.get('blocks_issued', 0)
            )
        
        with col4:
            st.metric(
                label="Active Services",
                value=honeypot_stats.get('active_services', 0),
                delta=0
            )
    
    def render_service_status(self):
        """Render service status"""
        st.subheader("🔧 Service Status")
        
        services_status = self.honeypot_manager.get_all_services_status()
        
        for service_name, status in services_status.items():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                status_icon = "🟢" if status.get('running', False) else "🔴"
                st.write(f"{status_icon} **{service_name.upper()}**")
            
            with col2:
                st.write(f"Port: {status.get('port', 'N/A')}")
            
            with col3:
                connections = status.get('total_connections', 0)
                st.write(f"Connections: {connections}")
            
            with col4:
                attacks = status.get('attack_detections', 0)
                st.write(f"Attacks: {attacks}")
    
    def render_threat_analysis(self):
        """Render threat analysis charts"""
        st.subheader("🚨 Threat Analysis")
        
        # Get threat data
        logs = self.honeypot_manager.get_all_logs(limit=1000)
        
        if logs:
            df = pd.DataFrame(logs)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Threats Over Time', 'Attack Types', 'Service Distribution', 'Source IPs'),
                specs=[[{"type": "scatter"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Threats over time
            if 'attack_type' in df.columns:
                attack_df = df[df['attack_type'].notna() & (df['attack_type'] != 'none')]
                if not attack_df.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=attack_df['timestamp'],
                            y=attack_df['confidence'],
                            mode='markers',
                            name='Threats',
                            marker=dict(color='red', size=8)
                        ),
                        row=1, col=1
                    )
            
            # Attack types pie chart
            if 'attack_type' in df.columns:
                attack_counts = df['attack_type'].value_counts()
                attack_counts = attack_counts[attack_counts.index != 'none']
                
                if not attack_counts.empty:
                    fig.add_trace(
                        go.Pie(
                            labels=attack_counts.index,
                            values=attack_counts.values,
                            name="Attack Types"
                        ),
                        row=1, col=2
                    )
            
            # Service distribution
            service_counts = df['service'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=service_counts.index,
                    y=service_counts.values,
                    name="Services"
                ),
                row=2, col=1
            )
            
            # Source IPs
            ip_counts = df['source_ip'].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=ip_counts.index,
                    y=ip_counts.values,
                    name="Source IPs"
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threat data available yet")
    
    def render_ml_insights(self):
        """Render ML model insights"""
        st.subheader("🤖 Machine Learning Insights")
        
        ml_stats = self.threat_detector.get_detection_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Status")
            
            # Anomaly detector status
            anomaly_status = ml_stats.get('anomaly_detector_trained', False)
            anomaly_icon = "🟢" if anomaly_status else "🔴"
            st.write(f"{anomaly_icon} Anomaly Detector: {'Trained' if anomaly_status else 'Not Trained'}")
            
            # Attack classifier status
            classifier_status = ml_stats.get('attack_classifier_trained', False)
            classifier_icon = "🟢" if classifier_status else "🔴"
            st.write(f"{classifier_icon} Attack Classifier: {'Trained' if classifier_status else 'Not Trained'}")
            
            # Detection statistics
            st.markdown("#### Detection Statistics")
            st.write(f"Total Detections: {ml_stats.get('total_detections', 0)}")
            st.write(f"Anomaly Detections: {ml_stats.get('anomaly_detections', 0)}")
            st.write(f"Classification Detections: {ml_stats.get('classification_detections', 0)}")
        
        with col2:
            st.markdown("#### Model Performance")
            
            # Create performance chart
            if ml_stats.get('total_detections', 0) > 0:
                detection_types = ['Anomaly', 'Classification']
                detection_counts = [
                    ml_stats.get('anomaly_detections', 0),
                    ml_stats.get('classification_detections', 0)
                ]
                
                fig = go.Figure(data=[
                    go.Bar(x=detection_types, y=detection_counts)
                ])
                fig.update_layout(title="Detection Types")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detection data available")
    
    def render_adaptive_response(self):
        """Render adaptive response information"""
        st.subheader("🛡️ Adaptive Response")
        
        response_stats = self.adaptive_response.get_status()
        blocked_ips = self.adaptive_response.get_blocked_ips()
        throttled_ips = self.adaptive_response.get_throttled_ips()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Response Statistics")
            st.write(f"Total Threats: {response_stats.get('total_threats', 0)}")
            st.write(f"Blocks Issued: {response_stats.get('blocks_issued', 0)}")
            st.write(f"Throttles Issued: {response_stats.get('throttles_issued', 0)}")
            st.write(f"Decoys Sent: {response_stats.get('decoys_sent', 0)}")
        
        with col2:
            st.markdown("#### Current Status")
            st.write(f"Blocked IPs: {response_stats.get('blocked_ips_count', 0)}")
            st.write(f"Throttled IPs: {response_stats.get('throttled_ips_count', 0)}")
            st.write(f"Threat History: {response_stats.get('threat_history_count', 0)}")
        
        # Blocked IPs table
        if blocked_ips:
            st.markdown("#### Blocked IP Addresses")
            blocked_df = pd.DataFrame([
                {
                    'IP': ip,
                    'Reason': data.get('reason', 'Unknown'),
                    'Blocked Until': data.get('block_until', 'Unknown'),
                    'Timestamp': data.get('timestamp', 'Unknown')
                }
                for ip, data in blocked_ips.items()
            ])
            st.dataframe(blocked_df, use_container_width=True)
    
    def render_recent_activity(self):
        """Render recent activity log"""
        st.subheader("📝 Recent Activity")
        
        logs = self.honeypot_manager.get_all_logs(limit=50)
        
        if logs:
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            # Display recent connections
            for _, log in df.head(10).iterrows():
                timestamp = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                source_ip = log.get('source_ip', 'Unknown')
                service = log.get('service', 'Unknown')
                attack_type = log.get('attack_type', 'none')
                
                if attack_type != 'none':
                    st.markdown(f"🚨 **{timestamp}** - Attack detected from {source_ip} on {service}: {attack_type}")
                else:
                    st.markdown(f"📡 **{timestamp}** - Connection from {source_ip} on {service}")
        else:
            st.info("No recent activity")
    
    def render_system_health(self):
        """Render system health indicators"""
        st.subheader("💚 System Health")
        
        # Get system status
        honeypot_stats = self.honeypot_manager.get_overall_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Uptime
            uptime = honeypot_stats.get('uptime', 0)
            uptime_hours = uptime / 3600
            st.metric("System Uptime", f"{uptime_hours:.1f} hours")
        
        with col2:
            # Active services
            active_services = honeypot_stats.get('active_services', 0)
            total_services = honeypot_stats.get('total_services', 0)
            health_percentage = (active_services / total_services * 100) if total_services > 0 else 0
            st.metric("Service Health", f"{health_percentage:.1f}%")
        
        with col3:
            # Response time (simulated)
            response_time = np.random.uniform(10, 100)  # Simulated
            st.metric("Avg Response Time", f"{response_time:.1f}ms")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        self.render_sidebar()
        
        # Check if it's time to refresh
        time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
        if time_since_update >= st.session_state.refresh_rate:
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Main content
        self.render_overview_metrics()
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 Services", "🚨 Threats", "🤖 ML Insights", "🛡️ Response", "📊 Health"
        ])
        
        with tab1:
            self.render_service_status()
        
        with tab2:
            self.render_threat_analysis()
        
        with tab3:
            self.render_ml_insights()
        
        with tab4:
            self.render_adaptive_response()
        
        with tab5:
            self.render_system_health()
        
        # Recent activity at the bottom
        self.render_recent_activity()

def main():
    """Main function"""
    try:
        dashboard = HoneypotAIDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.info("Make sure the HoneypotAI system is running")

if __name__ == "__main__":
    main()
