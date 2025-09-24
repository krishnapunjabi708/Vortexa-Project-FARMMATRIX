import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import math
import random

# Set page config for a clean and wide layout
st.set_page_config(
    page_title="AgriMarket Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2E7D32, #4CAF50);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4CAF50;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.profit-positive {
    color: #2E7D32;
    font-weight: bold;
}
.profit-negative {
    color: #d32f2f;
    font-weight: bold;
}
.section-header {
    color: #2E7D32;
    font-size: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 0 24px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
.search-bar {
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Sample data generation functions
@st.cache_data
def generate_market_data():
    """Generate comprehensive market data with 50+ mandis around Pune"""
    markets = [
        {"name": "Pune APMC Market", "location": "Pune", "distance": 5, "lat": 18.5204, "lon": 73.8567},
        {"name": "Hadapsar Mandi", "location": "Hadapsar, Pune", "distance": 12, "lat": 18.5089, "lon": 73.9260},
        {"name": "Kothrud Market", "location": "Kothrud, Pune", "distance": 8, "lat": 18.5074, "lon": 73.8077},
        {"name": "Chinchwad APMC", "location": "Chinchwad, Pune", "distance": 18, "lat": 18.6298, "lon": 73.8131},
        {"name": "Wakad Wholesale Market", "location": "Wakad, Pune", "distance": 20, "lat": 18.5975, "lon": 73.7898},
        {"name": "Baramati APMC", "location": "Baramati", "distance": 68, "lat": 18.1514, "lon": 74.5815},
        {"name": "Daund Market", "location": "Daund", "distance": 72, "lat": 18.4648, "lon": 74.5821},
        {"name": "Indapur Mandi", "location": "Indapur", "distance": 85, "lat": 18.1239, "lon": 75.0197},
        {"name": "Manoj Market Pune", "location": "Manoj, Pune", "distance": 15, "lat": 18.5314, "lon": 73.8446},
        {"name": "Saswad APMC", "location": "Saswad", "distance": 35, "lat": 18.3511, "lon": 74.0333},
        {"name": "Satara APMC", "location": "Satara", "distance": 108, "lat": 17.6805, "lon": 74.0183},
        {"name": "Karad Market", "location": "Karad", "distance": 125, "lat": 17.2893, "lon": 74.1822},
        {"name": "Koregaon Mandi", "location": "Koregaon", "distance": 95, "lat": 17.6167, "lon": 74.0500},
        {"name": "Wai APMC", "location": "Wai", "distance": 82, "lat": 17.9519, "lon": 73.8919},
        {"name": "Phaltan Market", "location": "Phaltan", "distance": 92, "lat": 17.9919, "lon": 74.4319},
        {"name": "Solapur APMC", "location": "Solapur", "distance": 142, "lat": 17.6599, "lon": 75.9064},
        {"name": "Pandharpur Mandi", "location": "Pandharpur", "distance": 118, "lat": 17.6794, "lon": 75.3189},
        {"name": "Akkalkot Market", "location": "Akkalkot", "distance": 165, "lat": 17.5244, "lon": 76.2067},
        {"name": "Barshi APMC", "location": "Barshi", "distance": 135, "lat": 18.2333, "lon": 75.6833},
        {"name": "Ahmednagar APMC", "location": "Ahmednagar", "distance": 118, "lat": 19.0948, "lon": 74.7480},
        {"name": "Shrirampur Market", "location": "Shrirampur", "distance": 78, "lat": 19.6167, "lon": 74.6667},
        {"name": "Sangamner APMC", "location": "Sangamner", "distance": 88, "lat": 19.5667, "lon": 74.2167},
        {"name": "Kopargaon Mandi", "location": "Kopargaon", "distance": 102, "lat": 19.8833, "lon": 74.4833},
        {"name": "Newasa Market", "location": "Newasa", "distance": 95, "lat": 19.6333, "lon": 74.9167},
        {"name": "Rahuri APMC", "location": "Rahuri", "distance": 108, "lat": 19.3833, "lon": 74.6500},
        {"name": "Nashik APMC", "location": "Nashik", "distance": 160, "lat": 19.9975, "lon": 73.7898},
        {"name": "Manmad Market", "location": "Manmad", "distance": 142, "lat": 20.2552, "lon": 74.4386},
        {"name": "Yeola APMC", "location": "Yeola", "distance": 125, "lat": 20.0422, "lon": 74.4889},
        {"name": "Niphad Mandi", "location": "Niphad", "distance": 138, "lat": 20.0833, "lon": 74.1167},
        {"name": "Sinnar Market", "location": "Sinnar", "distance": 145, "lat": 19.8500, "lon": 73.9833},
        {"name": "Kolhapur APMC", "location": "Kolhapur", "distance": 238, "lat": 16.7050, "lon": 74.2433},
        {"name": "Ichalkaranji Market", "location": "Ichalkaranji", "distance": 245, "lat": 16.6917, "lon": 74.4583},
        {"name": "Sangli APMC", "location": "Sangli", "distance": 208, "lat": 16.8524, "lon": 74.5815},
        {"name": "Miraj Market", "location": "Miraj", "distance": 218, "lat": 16.8276, "lon": 74.6266},
        {"name": "Aurangabad APMC", "location": "Aurangabad", "distance": 238, "lat": 19.8762, "lon": 75.3433},
        {"name": "Jalna Market", "location": "Jalna", "distance": 265, "lat": 19.8406, "lon": 75.8833},
        {"name": "Paithan APMC", "location": "Paithan", "distance": 225, "lat": 19.4833, "lon": 75.3833},
        {"name": "Gangapur Mandi", "location": "Gangapur", "distance": 248, "lat": 19.6967, "lon": 75.0100},
        {"name": "Mumbai APMC Vashi", "location": "Mumbai", "distance": 148, "lat": 19.0760, "lon": 72.8777},
        {"name": "Thane Market", "location": "Thane", "distance": 125, "lat": 19.2183, "lon": 72.9781},
        {"name": "Kalyan APMC", "location": "Kalyan", "distance": 112, "lat": 19.2437, "lon": 73.1355},
        {"name": "Panvel Market", "location": "Panvel", "distance": 132, "lat": 18.9894, "lon": 73.1106},
        {"name": "Alibag APMC", "location": "Alibag", "distance": 158, "lat": 18.6411, "lon": 72.8719},
        {"name": "Pen Market", "location": "Pen", "distance": 142, "lat": 18.7364, "lon": 73.0969},
        {"name": "Mahad APMC", "location": "Mahad", "distance": 175, "lat": 18.0833, "lon": 73.4167},
        {"name": "Ratnagiri APMC", "location": "Ratnagiri", "distance": 332, "lat": 16.9944, "lon": 73.3000},
        {"name": "Chiplun Market", "location": "Chiplun", "distance": 278, "lat": 17.5333, "lon": 73.5167},
        {"name": "Surat APMC", "location": "Surat, Gujarat", "distance": 285, "lat": 21.1702, "lon": 72.8311},
        {"name": "Bharuch Market", "location": "Bharuch, Gujarat", "distance": 318, "lat": 21.7051, "lon": 72.9959},
        {"name": "Junnar APMC", "location": "Junnar", "distance": 95, "lat": 19.2167, "lon": 73.8833},
        {"name": "Ambegaon Market", "location": "Ambegaon", "distance": 68, "lat": 18.7167, "lon": 73.7167},
        {"name": "Khed APMC", "location": "Khed", "distance": 55, "lat": 18.9500, "lon": 73.4000},
        {"name": "Rajgurunagar Market", "location": "Rajgurunagar", "distance": 45, "lat": 18.8667, "lon": 73.7833},
        {"name": "Shirur APMC", "location": "Shirur", "distance": 58, "lat": 18.8272, "lon": 74.3706},
        {"name": "Ranjangaon Market", "location": "Ranjangaon", "distance": 52, "lat": 18.5833, "lon": 74.1500}
    ]
    
    crops = ["Rice", "Wheat", "Onion", "Potato", "Tomato", "Cotton", "Sugarcane", "Soybean"]
    
    market_data = []
    for market in markets:
        for crop in crops:
            base_price = {
                "Rice": 2500, "Wheat": 2200, "Onion": 1800, "Potato": 1200,
                "Tomato": 2800, "Cotton": 5500, "Sugarcane": 3200, "Soybean": 4200
            }[crop]
            
            current_price = base_price + random.randint(-300, 500)
            msp = base_price * 0.8
            
            market_data.append({
                "Market": market["name"],
                "Location": market["location"],
                "Distance (km)": market["distance"],
                "Crop": crop,
                "Current Price (₹/quintal)": current_price,
                "MSP (₹/quintal)": msp,
                "Price vs MSP": current_price - msp,
                "Lat": market["lat"],
                "Lon": market["lon"]
            })
    
    return pd.DataFrame(market_data)

@st.cache_data
def generate_price_trends():
    """Generate sample price trend data"""
    crops = ["Rice", "Wheat", "Onion", "Potato", "Tomato", "Cotton", "Sugarcane", "Soybean"]
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    trend_data = []
    for crop in crops:
        base_price = {
            "Rice": 2500, "Wheat": 2200, "Onion": 1800, "Potato": 1200,
            "Tomato": 2800, "Cotton": 5500, "Sugarcane": 3200, "Soybean": 4200
        }[crop]
        
        prices = []
        current_price = base_price
        for date in dates:
            change = random.randint(-50, 50)
            current_price = max(base_price * 0.7, current_price + change)
            prices.append(current_price)
            
            trend_data.append({
                "Date": date,
                "Crop": crop,
                "Price": current_price
            })
    
    return pd.DataFrame(trend_data)

def calculate_transport_cost(distance, vehicle_type):
    """Calculate transport cost based on distance and vehicle type"""
    rates = {
        "Tractor Trolley": 15,
        "Tempo": 12,
        "Truck (Small)": 18,
        "Truck (Large)": 25,
        "Mini Truck": 10
    }
    
    return distance * rates.get(vehicle_type, 15)

def get_weather_data(city):
    """Simulate weather data"""
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    return {
        "temperature": random.randint(20, 35),
        "humidity": random.randint(40, 80),
        "condition": random.choice(weather_conditions),
        "wind_speed": random.randint(5, 15)
    }

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AgriMarket Intelligence Platform</h1>
        <p>Smart Agricultural Market Analysis & Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    market_df = generate_market_data()
    trends_df = generate_price_trends()
    
    # Sidebar for location and filters
    st.sidebar.header("📍 Location & Filters")
    
    location_method = st.sidebar.radio("Select Location Method:", ["Manual Input", "GPS Simulation"])
    
    if location_method == "Manual Input":
        farmer_location = st.sidebar.text_input("Enter your location:", "Pune, Maharashtra")
    else:
        farmer_location = "GPS: Pune, Maharashtra (18.5204, 73.8567)"
    
    st.sidebar.success(f"📍 Current Location: {farmer_location}")
    
    max_distance = st.sidebar.slider("Maximum Distance (km):", 0, 150, 100)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🏪 Home", "📈 Price Trends", "🚛 Transport & Profit", "🌤️ Weather"])
    
    with tab1:
        st.header("Nearby Markets")
        
        # Search bar for crops
        st.subheader("🔍 Search Crops")
        search_term = st.text_input("Search for crops:", key="crop_search", placeholder="Enter crop name (e.g., Rice, Wheat)")
        
        if search_term:
            selected_crops = [crop for crop in market_df["Crop"].unique() if search_term.lower() in crop.lower()]
            if not selected_crops:
                st.warning("No crops found matching your search.")
        else:
            selected_crops = market_df["Crop"].unique().tolist()
        
        # Filter data
        filtered_df = market_df[
            (market_df["Crop"].isin(selected_crops)) & 
            (market_df["Distance (km)"] <= max_distance)
        ]
        
        # Display total markets
        total_markets = len(filtered_df["Market"].unique())
        st.info(f"📊 Showing {total_markets} markets within {max_distance}km radius")
        
        # Nearest markets
        st.markdown('<div class="section-header">📍 Nearest Markets</div>', unsafe_allow_html=True)
        nearest_markets = filtered_df.groupby("Market")["Distance (km)"].first().sort_values().head(5)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for market in nearest_markets.index:
                market_data = filtered_df[filtered_df["Market"] == market]
                distance = market_data["Distance (km)"].iloc[0]
                location = market_data["Location"].iloc[0]
                
                with st.expander(f"📍 {market} ({distance:.0f} km)"):
                    st.markdown(f"**Location**: {location}")
                    
                    # Crop-wise analysis for this market
                    crop_analysis = market_data.groupby('Crop').agg({
                        'Current Price (₹/quintal)': ['mean'],
                        'MSP (₹/quintal)': ['mean'],
                        'Price vs MSP': 'mean'
                    }).round(0)
                    
                    crop_analysis.columns = ['Current Price', 'MSP', 'Price vs MSP']
                    crop_analysis = crop_analysis.reset_index()
                    
                    # Color code the Price vs MSP column
                    def color_price_diff(val):
                        if val > 500:
                            return 'background-color: #c8e6c9; color: #2e7d32'
                        elif val > 0:
                            return 'background-color: #fff3e0; color: #f57c00'
                        else:
                            return 'background-color: #ffcdd2; color: #d32f2f'
                    
                    styled_table = crop_analysis.style.applymap(color_price_diff, subset=['Price vs MSP'])
                    st.dataframe(styled_table, use_container_width=True)
        
        with col2:
            # Top performing crops by average profit margin
            st.markdown('<div class="section-header">🎯 Top Crops by Profit</div>', unsafe_allow_html=True)
            top_crops_profit = filtered_df.groupby("Crop")["Price vs MSP"].mean().sort_values(ascending=False).head(5)
            
            for i, (crop, margin) in enumerate(top_crops_profit.items()):
                profit_class = "profit-positive" if margin > 0 else "profit-negative"
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                st.markdown(f"""
                <div class="metric-card">
                    {rank_emoji} <strong>{crop}</strong><br>
                    <span class="{profit_class}">₹{margin:.0f} above MSP</span>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
        
        # Market Map
        st.markdown('<div class="section-header">🗺️ Market Locations Map</div>', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            map_df = filtered_df.groupby(["Market", "Lat", "Lon", "Distance (km)", "Location"]).agg({
                "Current Price (₹/quintal)": "mean",
                "Crop": "count"
            }).reset_index()
            map_df.rename(columns={"Crop": "Crops Available"}, inplace=True)
            
            map_df['Size'] = map_df['Crops Available'] * 50
            
            fig_map = px.scatter_mapbox(
                map_df,
                lat="Lat",
                lon="Lon",
                size="Size",
                hover_name="Market",
                hover_data={
                    "Location": True,
                    "Distance (km)": ":.0f", 
                    "Current Price (₹/quintal)": ":.0f",
                    "Crops Available": True,
                    "Size": False
                },
                color="Distance (km)",
                color_continuous_scale="Viridis_r",
                mapbox_style="open-street-map",
                zoom=6,
                height=400,
                title="Markets by Distance and Crop Availability"
            )
            
            fig_map.add_trace(go.Scattermapbox(
                lat=[18.5204],
                lon=[73.8567],
                mode='markers',
                marker=dict(size=20, color='red', symbol='circle'),
                text=["📍 Your Location (Pune)"],
                hovertemplate="<b>Your Location</b><br>Pune, Maharashtra<extra></extra>",
                showlegend=False,
                name="Farmer Location"
            ))
            
            fig_map.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                mapbox=dict(
                    center=dict(lat=18.8, lon=74.2),
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        st.header("📈 Price Trends")
        
        trend_period = st.selectbox("Select Trend Period:", ["7 Days", "30 Days"])
        days = 7 if trend_period == "7 Days" else 30
        
        trend_data = trends_df[trends_df["Date"] >= (datetime.now() - timedelta(days=days))]
        trend_data = trend_data[trend_data["Crop"].isin(selected_crops)]
        
        if not trend_data.empty:
            fig_trend = px.line(
                trend_data,
                x="Date",
                y="Price",
                color="Crop",
                title=f"Price Trends - Last {trend_period}",
                labels={"Price": "Price (₹/quintal)", "Date": "Date"}
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_prices = filtered_df.groupby("Crop")["Current Price (₹/quintal)"].mean()
                fig_bar = px.bar(
                    x=current_prices.index,
                    y=current_prices.values,
                    title="Current Average Prices",
                    labels={"x": "Crop", "y": "Price (₹/quintal)"}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                msp_comparison = filtered_df.groupby("Crop")["Price vs MSP"].mean()
                fig_msp = px.bar(
                    x=msp_comparison.index,
                    y=msp_comparison.values,
                    title="Price vs MSP Comparison",
                    labels={"x": "Crop", "y": "Difference from MSP (₹)"},
                    color=msp_comparison.values,
                    color_continuous_scale=["red", "yellow", "green"]
                )
                st.plotly_chart(fig_msp, use_container_width=True)
    
    with tab3:
        st.header("🚛 Transport & Profit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Transport Cost Calculator</div>', unsafe_allow_html=True)
            
            selected_market = st.selectbox("Select Market:", filtered_df["Market"].unique())
            market_distance = filtered_df[filtered_df["Market"] == selected_market]["Distance (km)"].iloc[0]
            
            vehicle_type = st.selectbox("Select Vehicle Type:", ["Tractor Trolley", "Tempo", "Truck (Small)", "Truck (Large)", "Mini Truck"])
            transport_cost = calculate_transport_cost(market_distance, vehicle_type)
            
            st.info(f"🚛 Transport Cost: ₹{transport_cost:.0f}")
            st.info(f"📏 Distance: {market_distance} km")
            st.info(f"💰 Cost per km: ₹{transport_cost/market_distance:.0f}")
        
        with col2:
            st.markdown('<div class="section-header">Net Profit Estimation</div>', unsafe_allow_html=True)
            
            crop_for_profit = st.selectbox("Select Crop for Profit Analysis:", selected_crops)
            quantity = st.number_input("Quantity (quintals):", min_value=1, value=10)
            
            market_price = filtered_df[
                (filtered_df["Market"] == selected_market) & 
                (filtered_df["Crop"] == crop_for_profit)
            ]["Current Price (₹/quintal)"].iloc[0]
            
            gross_revenue = market_price * quantity
            total_transport_cost = transport_cost
            net_profit = gross_revenue - total_transport_cost
            profit_per_quintal = net_profit / quantity
            
            st.metric("Market Price", f"₹{market_price:.0f}/quintal")
            st.metric("Gross Revenue", f"₹{gross_revenue:.0f}")
            st.metric("Transport Cost", f"₹{total_transport_cost:.0f}")
            
            if net_profit > 0:
                st.success(f"✅ Net Profit: ₹{net_profit:.0f}")
                st.success(f"💰 Profit per quintal: ₹{profit_per_quintal:.0f}")
            else:
                st.error(f"❌ Net Loss: ₹{abs(net_profit):.0f}")
                st.error(f"💸 Loss per quintal: ₹{abs(profit_per_quintal):.0f}")
        
        st.markdown('<div class="section-header">📊 Profit Comparison Across Markets</div>', unsafe_allow_html=True)
        
        profit_data = []
        for market in filtered_df["Market"].unique():
            market_dist = filtered_df[filtered_df["Market"] == market]["Distance (km)"].iloc[0]
            transport_cost_market = calculate_transport_cost(market_dist, vehicle_type)
            
            for crop in selected_crops:
                crop_data = filtered_df[(filtered_df["Market"] == market) & (filtered_df["Crop"] == crop)]
                if not crop_data.empty:
                    price = crop_data["Current Price (₹/quintal)"].iloc[0]
                    profit = (price * quantity) - transport_cost_market
                    
                    profit_data.append({
                        "Market": market,
                        "Crop": crop,
                        "Net Profit": profit,
                        "Profit per Quintal": profit / quantity
                    })
        
        if profit_data:
            profit_df = pd.DataFrame(profit_data)
            
            fig_profit = px.bar(
                profit_df,
                x="Market",
                y="Net Profit",
                color="Crop",
                title="Net Profit Comparison Across Markets",
                barmode="group"
            )
            st.plotly_chart(fig_profit, use_container_width=True)
    
    with tab4:
        st.header("🌤️ Weather Forecast")
        
        col1, col2, col3 = st.columns(3)
        
        locations = ["Mumbai", "Pune", "Nashik"]
        
        for i, location in enumerate(locations):
            weather = get_weather_data(location)
            
            with [col1, col2, col3][i]:
                st.markdown(f'<div class="section-header">🌍 {location}</div>', unsafe_allow_html=True)
                
                st.metric("🌡️ Temperature", f"{weather['temperature']}°C")
                st.metric("💧 Humidity", f"{weather['humidity']}%")
                st.metric("💨 Wind Speed", f"{weather['wind_speed']} km/h")
                
                condition_emoji = {
                    "Sunny": "☀️",
                    "Cloudy": "☁️",
                    "Rainy": "🌧️",
                    "Partly Cloudy": "⛅"
                }
                
                st.info(f"{condition_emoji.get(weather['condition'], '🌤️')} {weather['condition']}")
        
        st.markdown('<div class="section-header">🌾 Agricultural Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = [
            "🌱 Current weather is favorable for harvesting operations",
            "💧 Adequate moisture levels for crop growth",
            "🚜 Good conditions for field preparations",
            "📈 Market demand expected to increase due to seasonal factors"
        ]
        
        for rec in recommendations:
            st.success(rec)
        
        st.markdown('<div class="section-header">🌦️ Weather Impact on Prices</div>', unsafe_allow_html=True)
        
        weather_impact = pd.DataFrame({
            "Crop": selected_crops,
            "Weather Impact": [random.choice(["Positive", "Neutral", "Negative"]) for _ in selected_crops],
            "Price Change (%)": [random.randint(-15, 20) for _ in selected_crops]
        })
        
        def color_impact(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            else:
                return 'color: orange'
        
        st.dataframe(
            weather_impact.style.applymap(color_impact, subset=['Price Change (%)']),
            use_container_width=True
        )

if __name__ == "__main__":
    main()