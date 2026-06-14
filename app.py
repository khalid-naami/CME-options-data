"""
Main Streamlit Application - Refactored and Modular
CME Equity Options Pricer with Advanced Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config.settings import config

# Import business logic
from src.models.pricing import OptionsPricingEngine
from src.data.rates import get_risk_free_rates
from src.data.providers import MarketDataProvider
from src.utils.cache import cached_stock_info, cached_options_data
from src.utils.formatters import DataFormatter
from src.utils.validators import DataValidator
from src.ui.components import StreamlitComponents

def initialize_app():
    """Initialize Streamlit app configuration"""
    st.set_page_config(
        page_title=config.ui.page_title,
        page_icon=config.ui.page_icon,
        layout=config.ui.layout,
        initial_sidebar_state="expanded"
    )
    # --- Premium Social Links CSS ---
    st.markdown("""
    <style>
    .do-banner{background:linear-gradient(135deg,#0d1a12 0%,#111 100%);border:1px solid rgba(0,255,136,0.2);border-radius:14px;padding:18px 22px;margin-bottom:16px;}
    .do-banner a.do-title{color:#00ff88!important;font-weight:800;font-size:1.05rem;text-decoration:none;}
    .do-banner .do-desc{color:#888;font-size:0.78rem;margin-top:4px;}
    .social-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
    .social-row a{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:6px 12px;text-decoration:none;color:#ccc!important;font-size:0.78rem;font-weight:500;transition:all 0.2s;}
    .social-row a:hover{background:rgba(0,255,136,0.1);border-color:rgba(0,255,136,0.3);color:#00ff88!important;}
    .social-row a img{width:16px;height:16px;object-fit:contain;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rates_provider' not in st.session_state:
        st.session_state.rates_provider = get_risk_free_rates()
    
    if 'market_provider' not in st.session_state:
        st.session_state.market_provider = MarketDataProvider()
    
    if 'pricing_engine' not in st.session_state:
        st.session_state.pricing_engine = OptionsPricingEngine()

def handle_volatility_override():
    """Handle volatility override from URL parameters or localStorage"""
    try:
        # Check URL parameters for volatility
        query_params = st.query_params
        if 'volatility' in query_params:
            vol_value = float(query_params['volatility'])
            if DataValidator.validate_volatility(vol_value / 100):
                if 'volatility_override' not in st.session_state:
                    st.session_state['volatility_override'] = vol_value
                    st.sidebar.success(f"Volatility value {vol_value:.2f}% set from URL")
    except (ValueError, TypeError):
        pass

def fetch_market_data(symbol: str):
    """Fetch and cache market data for symbol"""
    if not symbol:
        return None, None, None
    
    # Validate symbol
    if not DataValidator.validate_symbol(symbol):
        st.error(f"Invalid symbol format: {symbol}")
        return None, None, None
    
    with st.spinner(f"Fetching data for {symbol}..."):
        # Get stock info
        stock_info = cached_stock_info(symbol)
        
        # Get options data
        options_data, current_price = cached_options_data(symbol)
        
        return stock_info, options_data, current_price

def calculate_theoretical_prices(options_df: pd.DataFrame, pricing_controls: dict, 
                               current_price: float) -> pd.DataFrame:
    """Calculate theoretical prices and add to options dataframe"""
    if not pricing_controls['enable_pricing'] or options_df.empty:
        return options_df
    
    enhanced_df = options_df.copy()
    pricing_engine = st.session_state.pricing_engine
    
    # Prepare lists for bulk calculations
    theoretical_prices = []
    price_diffs = []
    price_diff_pcts = []
    delta_list = []
    gamma_list = []
    theta_list = []
    vega_list = []
    rho_list = []
    
    for _, row in enhanced_df.iterrows():
        try:
            # Calculate time to expiry
            if 'time_to_expiry' in row and pd.notna(row['time_to_expiry']):
                T = row['time_to_expiry']
            else:
                days_to_expiry = (pd.to_datetime(row['expiration']) - datetime.now()).days
                T = max(0, days_to_expiry / 365.25)
            
            # Use override volatility or implied volatility
            if pricing_controls['volatility_override'] and pricing_controls['volatility_override'] > 0:
                sigma = pricing_controls['volatility_override']
            elif 'impliedVolatility' in row and pd.notna(row['impliedVolatility']):
                sigma = row['impliedVolatility']
            else:
                sigma = 0.25  # Default 25%
            
            # Calculate theoretical price
            if row['option_type'].lower() == 'call':
                theoretical_price = pricing_engine.black_scholes_call(
                    current_price, row['strike'], T, pricing_controls['risk_free_rate'], sigma
                )
            else:
                theoretical_price = pricing_engine.black_scholes_put(
                    current_price, row['strike'], T, pricing_controls['risk_free_rate'], sigma
                )
            
            # Calculate price difference
            if 'lastPrice' in row and pd.notna(row['lastPrice']) and row['lastPrice'] > 0:
                price_diff = theoretical_price - row['lastPrice']
                price_diff_pct = (price_diff / row['lastPrice']) * 100
            else:
                price_diff = 0
                price_diff_pct = 0
            
            # Calculate Greeks
            greeks = pricing_engine.calculate_greeks(
                current_price, row['strike'], T, pricing_controls['risk_free_rate'], 
                sigma, row['option_type']
            )
            
            theoretical_prices.append(theoretical_price)
            price_diffs.append(price_diff)
            price_diff_pcts.append(price_diff_pct)
            delta_list.append(greeks['delta'])
            gamma_list.append(greeks['gamma'])
            theta_list.append(greeks['theta'])
            vega_list.append(greeks['vega'])
            rho_list.append(greeks['rho'])
            
        except Exception as e:
            # Append zeros for failed calculations
            theoretical_prices.append(0)
            price_diffs.append(0)
            price_diff_pcts.append(0)
            delta_list.append(0)
            gamma_list.append(0)
            theta_list.append(0)
            vega_list.append(0)
            rho_list.append(0)
    
    # Add calculated columns
    enhanced_df['theoretical_price'] = theoretical_prices
    enhanced_df['price_diff'] = price_diffs
    enhanced_df['price_diff_pct'] = price_diff_pcts
    
    # Add theoretical Greeks (separate from market Greeks if they exist)
    enhanced_df['delta_theoretical'] = delta_list
    enhanced_df['gamma_theoretical'] = gamma_list
    enhanced_df['theta_theoretical'] = theta_list
    enhanced_df['vega_theoretical'] = vega_list
    enhanced_df['rho_theoretical'] = rho_list
    
    # If market Greeks don't exist, use theoretical as primary
    if 'delta' not in enhanced_df.columns:
        enhanced_df['delta'] = delta_list
    if 'gamma' not in enhanced_df.columns:
        enhanced_df['gamma'] = gamma_list
    if 'theta' not in enhanced_df.columns:
        enhanced_df['theta'] = theta_list
    if 'vega' not in enhanced_df.columns:
        enhanced_df['vega'] = vega_list
    if 'rho' not in enhanced_df.columns:
        enhanced_df['rho'] = rho_list
    
    return enhanced_df

def main():
    """Main application function"""
    # Initialize app
    initialize_app()
    initialize_session_state()
    handle_volatility_override()
    
    # Render header
    StreamlitComponents.render_header()
    
    # --- Sidebar: Dashboard Options + Social Links ---
    st.sidebar.markdown("""
    <div class="do-banner">
        <div>📊 <a class="do-title" href="https://dashboardoptions.com/" target="_blank">Dashboard Options</a></div>
        <div class="do-desc">A leading platform in financial market analysis</div>
        <div class="social-row">
            <a href="https://khalidnaami.com/" target="_blank"><img src="https://cdn.simpleicons.org/googlechrome/ffffff" />Khalid Naami</a>
            <a href="https://github.com/khalid-naami" target="_blank"><img src="https://cdn.simpleicons.org/github/ffffff" />GitHub</a>
            <a href="https://x.com/DashboardOption" target="_blank"><img src="https://cdn.simpleicons.org/x/ffffff" />𝕏</a>
            <a href="https://www.linkedin.com/company/dashboard-options/" target="_blank"><img src="https://cdn.simpleicons.org/linkedin/ffffff" />LinkedIn</a>
            <a href="https://www.instagram.com/dashboardoptions/" target="_blank"><img src="https://cdn.simpleicons.org/instagram/ffffff" />Instagram</a>
            <a href="https://discord.gg/NRSzCYRzpJ" target="_blank"><img src="https://cdn.simpleicons.org/discord/ffffff" />Discord</a>
            <a href="https://t.me/dashboardoptions" target="_blank"><img src="https://cdn.simpleicons.org/telegram/ffffff" />Telegram</a>
            <a href="https://www.youtube.com/@DashboardOptions" target="_blank"><img src="https://cdn.simpleicons.org/youtube/ffffff" />YouTube</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar and get inputs
    sidebar_inputs = StreamlitComponents.render_sidebar(st.session_state.rates_provider)
    symbol = sidebar_inputs['symbol']
    
    # Fetch market data
    stock_info, options_data, current_price = fetch_market_data(symbol)
    
    # Render stock metrics
    st.markdown("---")
    if stock_info:
        StreamlitComponents.render_stock_metrics(stock_info)
    else:
        st.info("💡 Enter a symbol and the stock information will appear here")
    
    # Options Chain Section
    st.markdown("---")
    st.subheader(f"📋 Options Chain for {symbol}")
    
    # Fetch button centered
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Auto-fetch on symbol change for better UX
        if st.button("🔄 Fetch Options Data", type="primary", use_container_width=True) or (symbol != st.session_state.get('last_symbol', '')):
            st.session_state['last_symbol'] = symbol
            
            if options_data is not None and current_price is not None:
                st.session_state['options_data'] = options_data
                st.session_state['current_price'] = current_price
                st.session_state['symbol'] = symbol
                # Center-aligned success message
                st.markdown(
                    f'<div style="text-align: center;"><span style="color: #28a745; background-color: #d4edda; '
                    f'border: 1px solid #c3e6cb; border-radius: 0.375rem; padding: 0.75rem 1rem; '
                    f'display: inline-block; font-weight: 500;">✅ Loaded {len(options_data)} options contracts</span></div>',
                    unsafe_allow_html=True
                )
            else:
                # Center-aligned error message
                st.markdown(
                    f'<div style="text-align: center;"><span style="color: #721c24; background-color: #f8d7da; '
                    f'border: 1px solid #f5c6cb; border-radius: 0.375rem; padding: 0.75rem 1rem; '
                    f'display: inline-block; font-weight: 500;">❌ Error fetching options data for {symbol}</span></div>',
                    unsafe_allow_html=True
                )
    
    # Display options data if available
    if 'options_data' in st.session_state and st.session_state['options_data'] is not None:
        options_df = st.session_state['options_data']
        current_price = st.session_state['current_price']
        
        # Filter options
        st.markdown("---")
        filtered_df = StreamlitComponents.render_options_filters(options_df, current_price)
        
        # Apply strike filter if selected
        if 'selected_strike_filter' in st.session_state and st.session_state['selected_strike_filter'] is not None:
            selected_strike_value = st.session_state['selected_strike_filter']
            filtered_df = filtered_df[filtered_df['strike'] == selected_strike_value]
            st.info(f"🎯 Showing options for strike: ${selected_strike_value:.2f}")
        
        if len(filtered_df) > 0:
            # Show active strike filter info
            if 'selected_strike_filter' in st.session_state and st.session_state['selected_strike_filter'] is not None:
                selected_strike_value = st.session_state['selected_strike_filter']
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"🎯 Filtered to strike: ${selected_strike_value:.2f} ({len(filtered_df)} options)")
                with col2:
                    if st.button("Clear Filter", key="clear_strike_filter"):
                        st.session_state['selected_strike_filter'] = None
                        st.rerun()
            
            # Pricing analysis section
            st.markdown("---")
            pricing_controls = StreamlitComponents.render_pricing_controls(
                st.session_state.rates_provider, filtered_df
            )
            
            # Render rate suggestions
            if pricing_controls.get('rate_suggestions'):
                StreamlitComponents.render_rate_suggestions(pricing_controls['rate_suggestions'])
            
            # Calculate theoretical prices if enabled
            if pricing_controls['enable_pricing']:
                with st.spinner("Calculating theoretical prices..."):
                    enhanced_df = calculate_theoretical_prices(
                        filtered_df, pricing_controls, current_price
                    )
                
                # Show pricing breakdown for first option (example)
                if not enhanced_df.empty:
                    first_option = enhanced_df.iloc[0]
                    breakdown = st.session_state.pricing_engine.calculate_pricing_breakdown(
                        current_price,
                        first_option['strike'],
                        first_option.get('time_to_expiry', 0.25),
                        pricing_controls['risk_free_rate'],
                        pricing_controls['volatility_override'] or 0.25,
                        first_option['option_type']
                    )
                    StreamlitComponents.render_pricing_breakdown(breakdown)
            else:
                enhanced_df = filtered_df
            
            # Display options table
            st.markdown("---")
            st.subheader("📊 Market vs Theoretical Prices")
            StreamlitComponents.render_options_table(enhanced_df)
            
            # Add comprehensive option analysis plots
            st.markdown("---")
            StreamlitComponents.render_option_analysis_plots(
                enhanced_df, current_price, pricing_controls, st.session_state.rates_provider
            )
        
        else:
            st.warning("No options match the current filters")
    
    else:
        st.info("Click 'Fetch Options Data' to load options chain")

if __name__ == "__main__":
    main()
    # --- Footer ---
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px; color:#555; font-size:0.78rem; margin-top:40px; border-top:1px solid #1a1a1a;">
        <a href="https://dashboardoptions.com/" style="color:#00ff88; text-decoration:none; font-weight:700;">Dashboard Options</a> — A leading platform in financial market analysis<br>
        <span style="margin-top:6px; display:inline-block;">
            <a href="https://khalidnaami.com/" style="color:#666; text-decoration:none;">Khalid Naami</a> ·
            <a href="https://github.com/khalid-naami" style="color:#666; text-decoration:none;">GitHub</a> ·
            <a href="https://x.com/DashboardOption" style="color:#666; text-decoration:none;">𝕏</a> ·
            <a href="https://www.linkedin.com/company/dashboard-options/" style="color:#666; text-decoration:none;">LinkedIn</a> ·
            <a href="https://www.instagram.com/dashboardoptions/" style="color:#666; text-decoration:none;">Instagram</a> ·
            <a href="https://discord.gg/NRSzCYRzpJ" style="color:#666; text-decoration:none;">Discord</a> ·
            <a href="https://t.me/dashboardoptions" style="color:#666; text-decoration:none;">Telegram</a> ·
            <a href="https://www.youtube.com/@DashboardOptions" style="color:#666; text-decoration:none;">YouTube</a>
        </span>
    </div>
    """, unsafe_allow_html=True)
