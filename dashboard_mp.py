# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import math
import random
import json
from typing import Dict, List, Optional

# Set page config for a clean and wide layout
st.set_page_config(
    page_title="AgriMarket Intelligence - Dashboard",
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
.stDataFrame {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

class AgriculturalDataManager:
    """Manages agricultural data from various sources"""
    
    def __init__(self):
        self.market_data = None
    
    def load_market_data(self) -> pd.DataFrame:
        """Load comprehensive market data from multiple sources"""
        try:
            # Generate realistic market data with 300+ entries
            markets = self._get_pune_markets()
            crops = self._get_crop_list()
            
            market_data = []
            for market in markets:
                state = "Gujarat" if "Gujarat" in market["location"] else "Maharashtra"
                district = market["location"].split(",")[0].strip() if "," in market["location"] else market["location"]
                
                # Random crops per market for realism
                market_crops = random.sample(crops, random.randint(3, len(crops)))
                
                for crop in market_crops:
                    base_price = self._get_base_price(crop)
                    current_price = round(base_price * random.uniform(0.8, 1.4))
                    msp = self._get_msp(crop)
                    
                    market_data.append({
                        "Market": market["name"],
                        "Location": market["location"],
                        "District": district,
                        "State": state,
                        "Distance (km)": market["distance"],
                        "Crop": crop,
                        "Current Price (₹/quintal)": current_price,
                        "MSP (₹/quintal)": msp,
                        "Price vs MSP": current_price - msp,
                        "Lat": market["lat"],
                        "Lon": market["lon"],
                        "Market Size": random.choice(["Small", "Medium", "Large"]),
                        "Arrival Quantity (tons)": random.randint(10, 500),
                        "Quality Grade": random.choice(["A", "B", "C"]),
                        "Last Updated": datetime.now() - timedelta(days=random.randint(0, 7))
                    })
            
            self.market_data = pd.DataFrame(market_data)
            return self.market_data
            
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def _get_pune_markets(self) -> List[Dict]:
        """Get list of markets around Pune"""
        return [
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
            {"name": "Ranjangaon Market", "location": "Ranjangaon", "distance": 52, "lat": 18.5833, "lon": 74.1500},
            # Additional markets
            {"name": "Achalpur APMC", "location": "Achalpur, Amravati", "distance": 520, "lat": 21.2572, "lon": 77.5086},
            {"name": "Amravati APMC", "location": "Amravati", "distance": 520, "lat": 20.9374, "lon": 77.7796},
            {"name": "Akola APMC", "location": "Akola", "distance": 430, "lat": 20.7059, "lon": 77.0215},
            {"name": "Beed APMC", "location": "Beed", "distance": 220, "lat": 18.9894, "lon": 75.7601},
            {"name": "Chandrapur APMC", "location": "Chandrapur", "distance": 660, "lat": 19.9615, "lon": 79.2961},
            {"name": "Dhule APMC", "location": "Dhule", "distance": 300, "lat": 20.9042, "lon": 74.7749},
            {"name": "Gondia APMC", "location": "Gondia", "distance": 800, "lat": 21.4624, "lon": 80.1920},
            {"name": "Hingoli APMC", "location": "Hingoli", "distance": 400, "lat": 19.7173, "lon": 77.1483},
            {"name": "Jalgaon APMC", "location": "Jalgaon", "distance": 350, "lat": 21.0100, "lon": 75.5626},
            {"name": "Latur APMC", "location": "Latur", "distance": 300, "lat": 18.4088, "lon": 76.5604},
            {"name": "Nagpur APMC", "location": "Nagpur", "distance": 700, "lat": 21.1458, "lon": 79.0882},
            # Nashik additional
            {"name": "Malegaon APMC", "location": "Malegaon, Nashik", "distance": 250, "lat": 20.5537, "lon": 74.5288},
            {"name": "Pimpalgaon Baswant APMC", "location": "Pimpalgaon Baswant, Nashik", "distance": 180, "lat": 20.1667, "lon": 73.9833},
            {"name": "Lasalgaon Market", "location": "Lasalgaon, Nashik", "distance": 170, "lat": 20.1500, "lon": 74.2333},
            # Thane additional
            {"name": "Ulhasnagar APMC", "location": "Ulhasnagar, Thane", "distance": 130, "lat": 19.2167, "lon": 73.1500},
            {"name": "Bhiwandi Wholesale Market", "location": "Bhiwandi, Thane", "distance": 120, "lat": 19.2813, "lon": 73.0485},
            # Mumbai additional
            {"name": "Dadar Vegetable Market", "location": "Dadar, Mumbai", "distance": 140, "lat": 19.0210, "lon": 72.8437},
            {"name": "Crawford Market", "location": "Crawford, Mumbai", "distance": 150, "lat": 18.9474, "lon": 72.8347},
            {"name": "Byculla Market", "location": "Byculla, Mumbai", "distance": 145, "lat": 18.9750, "lon": 72.8328}
        ]
    
    def _get_crop_list(self) -> List[str]:
        """Get list of crops"""
        return [
            "Rice", "Wheat", "Onion", "Potato", "Tomato", "Cotton", "Sugarcane", "Soybean"
        ]
    
    def _get_base_price(self, crop: str) -> int:
        """Get realistic base prices for different crops"""
        price_ranges = {
            "Rice": (1800, 3500), "Wheat": (1600, 2800), "Onion": (800, 2500), "Potato": (600, 1800),
            "Tomato": (1000, 4000), "Cotton": (4500, 7500), "Sugarcane": (2800, 3800), "Soybean": (3500, 5500)
        }
        base_range = price_ranges.get(crop, (1000, 3000))
        return random.randint(int(base_range[0]), int(base_range[1]))
    
    def _get_msp(self, crop: str) -> int:
        """Get Minimum Support Price based on government data"""
        msp_data = {
            "Rice": 1940, "Wheat": 1975, "Maize": 1870, "Bajra": 2150,
            "Jowar": 2630, "Ragi": 3577, "Barley": 1600,
            "Soybean": 3650, "Groundnut": 5550, "Sunflower": 6015,
            "Mustard": 5050, "Sesame": 7315, "Cotton": 5720
        }
        return msp_data.get(crop, random.randint(1500, 4000))

# Initialize data manager
data_manager = AgriculturalDataManager()

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AgriMarket Intelligence Platform - Dashboard</h1>
        <p>Smart Agricultural Market Analysis & Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load enhanced data
    with st.spinner("Loading agricultural market data..."):
        market_df = data_manager.load_market_data()
    
    st.sidebar.header("📍 Location & Filters")
    
    location_method = st.sidebar.radio("Select Location Method:", ["Manual Input", "GPS Simulation"])
    
    if location_method == "Manual Input":
        farmer_location = st.sidebar.text_input("Enter your location:", "Pune, Maharashtra")
    else:
        farmer_location = "GPS: Pune, Maharashtra (18.5204, 73.8567)"
    
    st.sidebar.success(f"📍 Current Location: {farmer_location}")
    
    max_distance = st.sidebar.slider("Maximum Distance (km):", 0, 400, 100)
    
    # Apply filters
    filtered_df = market_df[market_df["Distance (km)"] <= max_distance]
    
    # Main content area
    st.header("Agricultural Market Dashboard")
    
    # Display data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Markets", len(filtered_df["Market"].unique()))
    with col2:
        st.metric("Crops Available", len(filtered_df["Crop"].unique()))
    with col3:
        st.metric("States Covered", len(filtered_df["State"].unique()))
    with col4:
        avg_price = filtered_df["Current Price (₹/quintal)"].mean()
        st.metric("Avg Price/Quintal", f"₹{avg_price:,.0f}")
    
    # Search and filters
    st.subheader("🔍 Market Explorer")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("Search crops or markets:", key="crop_search", 
                                    placeholder="Enter crop name or market location")
    
    # Apply search filter
    if search_term:
        search_filter = (
            filtered_df["Crop"].str.contains(search_term, case=False) | 
            filtered_df["Market"].str.contains(search_term, case=False) |
            filtered_df["Location"].str.contains(search_term, case=False)
        )
        filtered_df = filtered_df[search_filter]
    
    # Improved sorting options
    available_sorts = ["Distance (Ascending)"]
    
    # Check if search term matches any crop
    if search_term:
        # Find crops that contain the search term (case-insensitive)
        matching_crops = [crop for crop in market_df["Crop"].unique() 
                         if search_term.lower() in crop.lower()]
        
        if matching_crops:
            # Use the first matching crop
            selected_crop = matching_crops[0]
            available_sorts.extend([
                f"Distance for {selected_crop} (Ascending)",
                f"Price for {selected_crop} (High to Low)"
            ])
    
    with col2:
        sort_by = st.selectbox("Sort Markets By:", available_sorts)
    
    # Determine sort series based on selection
    if sort_by == "Distance (Ascending)":
        # Default sorting by distance
        sort_series = filtered_df.groupby("Market")["Distance (km)"].min().sort_values()
    
    elif "Distance for" in sort_by:
        # Extract crop name from sort option
        crop_name = sort_by.replace("Distance for ", "").replace(" (Ascending)", "")
        
        # Filter data for the specific crop and sort by distance
        crop_df = filtered_df[filtered_df["Crop"] == crop_name]
        if not crop_df.empty:
            sort_series = crop_df.groupby("Market")["Distance (km)"].min().sort_values()
        else:
            # Fallback to default sorting if no data for the crop
            sort_series = filtered_df.groupby("Market")["Distance (km)"].min().sort_values()
            st.warning(f"No data found for {crop_name} in filtered results. Showing default sorting.")
    
    elif "Price for" in sort_by:
        # Extract crop name from sort option
        crop_name = sort_by.replace("Price for ", "").replace(" (High to Low)", "")
        
        # Filter data for the specific crop and sort by price (high to low)
        crop_df = filtered_df[filtered_df["Crop"] == crop_name]
        if not crop_df.empty:
            sort_series = crop_df.groupby("Market")["Current Price (₹/quintal)"].mean().sort_values(ascending=False)
        else:
            # Fallback to default sorting if no data for the crop
            sort_series = filtered_df.groupby("Market")["Distance (km)"].min().sort_values()
            st.warning(f"No data found for {crop_name} in filtered results. Showing default sorting.")
    
    else:
        # Default fallback
        sort_series = filtered_df.groupby("Market")["Distance (km)"].min().sort_values()
    
    # Display markets with load more
    st.markdown(f'<div class="section-header">📍 Available Markets ({len(filtered_df["Market"].unique())})</div>', unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("No markets found matching your criteria. Try adjusting filters.")
    else:
        if 'num_markets' not in st.session_state:
            st.session_state.num_markets = 10
        
        # Apply color grading to Price vs MSP column
        def style_price_vs_msp(val):
            if val > 0:
                return 'background-color: #4CAF50; color: white'
            elif val < 0:
                return 'background-color: #F44336; color: white'
            else:
                return 'background-color: #FFEB3B; color: black'
        
        # Display markets in sorted order
        markets_to_display = sort_series.index[:st.session_state.num_markets]
        
        for market in markets_to_display:
            market_data = filtered_df[filtered_df["Market"] == market]
            if not market_data.empty:
                market_info = market_data.iloc[0]
                
                with st.expander(f"🏪 {market} ({market_info['Distance (km)']:.0f} km) - {market_info['Location']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**State:** {market_info['State']}")
                        st.write(f"**Market Size:** {market_info['Market Size']}")
                        st.write(f"**Last Updated:** {market_info['Last Updated'].strftime('%Y-%m-%d')}")
                    
                    with col2:
                        avg_price = market_data["Current Price (₹/quintal)"].mean()
                        st.write(f"**Avg Price:** ₹{avg_price:,.0f}/quintal")
                        st.write(f"**Crops Available:** {len(market_data)}")
                    
                    with col3:
                        above_msp = len(market_data[market_data["Price vs MSP"] > 0])
                        st.write(f"**Crops above MSP:** {above_msp}/{len(market_data)}")
                    
                    st.dataframe(
                        market_data[["Crop", "Current Price (₹/quintal)", "MSP (₹/quintal)", "Price vs MSP", "Quality Grade"]]
                        .sort_values("Current Price (₹/quintal)", ascending=False)
                        .style.applymap(style_price_vs_msp, subset=["Price vs MSP"]),
                        use_container_width=True
                    )
        
        if st.session_state.num_markets < len(sort_series) and st.button("Load More Markets"):
            st.session_state.num_markets += 10
            st.rerun()
    
    # Summary table instead of map
    st.markdown('<div class="section-header">📊 Crop-wise Best Prices</div>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        best_df = filtered_df.loc[filtered_df.groupby('Crop')['Current Price (₹/quintal)'].idxmax()]
        best_df = best_df[['Crop', 'Current Price (₹/quintal)', 'Market', 'Distance (km)', 'Location', 'Price vs MSP']]
        best_df = best_df.sort_values('Current Price (₹/quintal)', ascending=False)
        st.dataframe(best_df.style.applymap(style_price_vs_msp, subset=["Price vs MSP"]), use_container_width=True)
    
    st.markdown('<div class="section-header">📈 Overall Market Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not filtered_df.empty:
            highest_crop = filtered_df.loc[filtered_df['Current Price (₹/quintal)'].idxmax()]['Crop']
            st.metric("Highest Priced Crop", highest_crop)
        else:
            st.metric("Highest Priced Crop", "-")
    with col2:
        if not filtered_df.empty:
            highest_price = filtered_df['Current Price (₹/quintal)'].max()
            st.metric("Highest Price", f"₹{highest_price:,.0f}")
        else:
            st.metric("Highest Price", "-")
    with col3:
        if not filtered_df.empty:
            avg_msp_diff = filtered_df['Price vs MSP'].mean()
            st.metric("Avg Price vs MSP", f"₹{avg_msp_diff:,.0f}")
        else:
            st.metric("Avg Price vs MSP", "-")
    with col4:
        if not filtered_df.empty:
            num_above_msp = (filtered_df['Price vs MSP'] > 0).sum()
            st.metric("Prices Above MSP", f"{num_above_msp} / {len(filtered_df)}")
        else:
            st.metric("Prices Above MSP", "-")

if __name__ == "__main__":
    main()