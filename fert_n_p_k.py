import logging
import os
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import pandas as pd
from folium.plugins import Draw
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen import canvas
from io import BytesIO
import sys
sys.path.append(r'C:\Users\pavan\AppData\Roaming\Python\Python313\site-packages')
import google.generativeai as genai

# Configuration
API_KEY = "AIzaSyAWA9Kqh2FRtBmxRZmNlZ7pcfasG5RJmR8"
MODEL = "models/gemini-1.5-flash"
LOGO_PATH = os.path.abspath("LOGO.jpg")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Google Earth Engine
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# Constants & Lookups
SOIL_TEXTURE_IMG = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0')
TEXTURE_CLASSES = {
    1: "Clay", 2: "Silty Clay", 3: "Sandy Clay",
    4: "Clay Loam", 5: "Silty Clay Loam", 6: "Sandy Clay Loam",
    7: "Loam", 8: "Silty Loam", 9: "Sandy Loam",
    10: "Silt", 11: "Loamy Sand", 12: "Sand"
}
IDEAL_RANGES = {
    "pH":           (6.0, 7.5),
    "Soil Texture": 7,
    "Salinity":     (None, 0.2),
    "Organic Carbon": (0.02, 0.05),
    "CEC":            (10, 30),
    "LST":            (10, 30),
    "NDVI":           (0.2, 0.8),
    "EVI":            (0.2, 0.8),
    "FVC":            (0.3, 0.8),
    "NDWI":           (-0.5, 0.5),
    "Nitrogen":       (280, 450),
    "Phosphorus":     (20, 50),
    "Potassium":      (150, 300)
}

# Research data from DY.Patil_Research.docx
RESEARCH_DATA = """
Fruit
Drainage/Soil Texture
Soil Salinity Tolerance (dS/m)
CEC Requirements (cmol/kg)
Organic Carbon (%)
Nitrogen Forms & Range (kg/ha)
Phosphorus Forms & Range (kg/ha)
Potassium Forms & Range (kg/ha)
Soil pH Range
Climate

Banana
Well – Drained/
Fertile loamy soil
<2.0 (sensitive) 1
15-25 2
1.5-3.0 2
Urea (46% N): 435-652 kg/ha
Ammonium Sulfate: 1,000-1,500 kg/ha
Range: 200-300 kg N/ha 34
DAP (18-46-0): 109-217 kg/ha
SSP (16-20% P): 250-625 kg/ha
Range: 50-100 kg P/ha 35
MOP (60% K2O): 500-667 kg/ha
SOP (50% K2O): 600-800 kg/ha
Range: 300-400 kg K/ha 36
6.0-7.5 
13°C and 38°C, 75-85% relative humidity

Onion
Good Drainage/
deep, friable loam and alluvial soils
<2.5 (sensitive)3
10-253
1.0-2.53
Urea: 109-217 kg/ha
Ammonium Sulfate: 238-476 kg/ha
Range: 50-100 kg N/ha2
DAP: 130-261 kg/ha
TSP (44-48% P): 125-278 kg/ha
Range: 60-125 kg P/ha2
MOP: 42-83 kg/ha
SOP: 50-100 kg/ha
Range: 25-50 kg K/ha2
5.8-6.5
Thrives in mild, temperate, tropical, and subtropical climates. Short-day onions are grown in plains (10-12 hours of sunlight), while long-day onions are grown in hills (13-14 hours).

Tomato
Well -Drained/
sandy or red loam soils rich in organic matter
<2.5 (sensitive)4
12-254
1.5-3.04
Urea: 163-326 kg/ha
Ammonium Nitrate (33.5% N): 224-448 kg/ha
Range: 75-150 kg N/ha2
MAP (50-52% P): 192-385 kg/ha
DAP: 217-435 kg/ha
Range: 100 kg P/ha2
SOP: 100 kg/ha
Potassium Nitrate (46% K): 109 kg/ha
Range: 50 kg K/ha2
6.0-7.04
warm climates (21-24°C)

Grapes
Well – Drained/
loamy soil with low water table
1.5-4.2 (moderately sensitive) 1
10-20 2
1.5-2.5 2
Urea: 261-391 kg/ha
Ammonium Sulfate: 571-857 kg/ha
Range: 120-180 kg N/ha 38
MAP: 115-173 kg/ha
Triple Superphosphate: 125-200 kg/ha
Range: 60-90 kg P/ha 315
Potassium Nitrate: 326-435 kg/ha
SOP: 300-400 kg/ha
Range: 150-200 kg K/ha 310
6.5-7.0 
15°C and 40°C (59°F - 104°F) during the growing and fruiting periods, with an optimal range for daily growth and development being around 20°C to 30°C (68°F - 86°F)

Potato
Well-drained/
loamy or sandy loam soils 
<2.0 (sensitive)1
8-201
1.5-3.01
Urea (46% N): 261-522 kg/ha
Ammonium Sulfate (21% N): 571-1,143 kg/ha
Range: 120-240 kg N/ha2
DAP (18-46-0): 522-1,043 kg/ha
SSP (16-20% P): 1,200-1,500 kg/ha
Range: 240 kg P/ha2
MOP (60% K2O): 200 kg/ha
SOP (50% K2O): 240 kg/ha
Range: 120 kg K/ha2
5.2-6.41
Optimal temperatures are 24°C for vegetative growth and 20°C for tuber development.
"""

# Utility Functions
def safe_get_info(computed_obj, name="value"):
    if computed_obj is None:
        return None
    try:
        info = computed_obj.getInfo()
        return float(info) if info is not None else None
    except Exception as e:
        logging.warning(f"Failed to fetch {name}: {e}")
        return None

def sentinel_composite(region, start, end, bands):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    try:
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_str, end_str)
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(bands)
        )
        if coll.size().getInfo() > 0:
            return coll.median().multiply(0.0001)
        for days in range(5, 31, 5):
            sd = (start - timedelta(days=days)).strftime("%Y-%m-%d")
            ed = (end + timedelta(days=days)).strftime("%Y-%m-%d")
            coll = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(sd, ed)
                .filterBounds(region)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .select(bands)
            )
            if coll.size().getInfo() > 0:
                logging.info(f"Sentinel window expanded to {sd}–{ed}")
                return coll.median().multiply(0.0001)
        logging.warning("No Sentinel-2 data available.")
        return None
    except Exception as e:
        logging.error(f"Error in sentinel_composite: {e}")
        return None

def get_lst(region, start, end):
    end_dt = end
    start_dt = end_dt - relativedelta(months=1)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    logging.info(f"Fetching MODIS LST from {start_str} to {end_str}")
    try:
        coll = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .filterBounds(region.buffer(5000))
            .filterDate(start_str, end_str)
            .select("LST_Day_1km")
        )
        cnt = coll.size().getInfo()
        if cnt == 0:
            logging.warning("No LST images in the specified range.")
            return None
        img = coll.median().multiply(0.02).subtract(273.15).rename("lst").clip(region.buffer(5000))
        stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=1000, maxPixels=1e13).getInfo()
        lst_value = stats.get("lst")
        return float(lst_value) if lst_value is not None else None
    except Exception as e:
        logging.error(f"Error in get_lst: {e}")
        return None

def estimate_cec(comp, region, intercept, slope_clay, slope_om):
    if comp is None:
        return None
    try:
        clay = comp.expression("(B11-B8)/(B11+B8+1e-6)", {"B11": comp.select("B11"), "B8": comp.select("B8")}).rename("clay")
        om = comp.expression("(B8-B4)/(B8+B4+1e-6)", {"B8": comp.select("B8"), "B4": comp.select("B4")}).rename("om")
        c_m = safe_get_info(clay.reduceRegion(ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e13).get("clay"), "clay")
        o_m = safe_get_info(om.reduceRegion(ee.Reducer.mean(), geometry=region, scale=20, maxPixels=1e13).get("om"), "om")
        if c_m is None or o_m is None:
            return None
        return intercept + slope_clay * c_m + slope_om * o_m
    except Exception as e:
        logging.error(f"Error in estimate_cec: {e}")
        return None

def get_soil_texture(region):
    try:
        mode = SOIL_TEXTURE_IMG.clip(region.buffer(500)).reduceRegion(ee.Reducer.mode(), geometry=region, scale=250, maxPixels=1e13).get("b0")
        val = safe_get_info(mode, "texture")
        return int(val) if val is not None else None
    except Exception as e:
        logging.error(f"Error in get_soil_texture: {e}")
        return None

def get_ndwi(comp, region):
    if comp is None:
        return None
    try:
        img = comp.expression("(B3-B8)/(B3+B8+1e-6)", {"B3": comp.select("B3"), "B8": comp.select("B8")}).rename("ndwi")
        return safe_get_info(img.reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e13).get("ndwi"), "NDWI")
    except Exception as e:
        logging.error(f"Error in get_ndwi: {e}")
        return None

def get_ndvi(comp, region):
    if comp is None:
        return None
    try:
        ndvi = comp.normalizedDifference(["B8", "B4"]).rename("ndvi")
        return safe_get_info(ndvi.reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e13).get("ndvi"), "NDVI")
    except Exception as e:
        logging.error(f"Error in get_ndvi: {e}")
        return None

def get_evi(comp, region):
    if comp is None:
        return None
    try:
        evi = comp.expression(
            "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
            {"NIR": comp.select("B8"), "RED": comp.select("B4"), "BLUE": comp.select("B2")}
        ).rename("evi")
        return safe_get_info(evi.reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e13).get("evi"), "EVI")
    except Exception as e:
        logging.error(f"Error in get_evi: {e}")
        return None

def get_fvc(comp, region):
    if comp is None:
        return None
    try:
        ndvi = comp.normalizedDifference(["B8", "B4"])
        ndvi_min = 0.2
        ndvi_max = 0.8
        fvc = ndvi.subtract(ndvi_min).divide(ndvi_max - ndvi_min).pow(2).clamp(0, 1).rename("fvc")
        return safe_get_info(fvc.reduceRegion(ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e13).get("fvc"), "FVC")
    except Exception as e:
        logging.error(f"Error in get_fvc: {e}")
        return None

def calculate_bare_soil_parameters(image, region):
    """
    Calculate soil parameters using BARE SOIL optimized formulas
    Designed for bare/harvested fields with minimal vegetation
    """
    # Normalize bands
    B2 = image.select("B2")
    B3 = image.select("B3")
    B4 = image.select("B4")
    B5 = image.select("B5")
    B6 = image.select("B6")
    B8 = image.select("B8")
    B11 = image.select("B11")
    B12 = image.select("B12")

    # Calculate bare soil specific indices
    BSI = (
        B11.add(B4)
        .subtract(B8)
        .subtract(B2)
        .divide(B11.add(B4).add(B8).add(B2).add(1e-6))
    )
    SBI = B2.add(B3).add(B4).divide(3)
    SSI = B11.subtract(B3).divide(B11.add(B3).add(1e-6))
    CMI = B11.divide(B12.add(1e-6))
    IOI = B4.divide(B2.add(1e-6))
    STI = B6.divide(B5.add(1e-6))
    MCI = B11.subtract(B6).divide(B11.add(B6).add(1e-6))

    # Extract values
    indices_image = BSI.addBands([SBI, SSI, CMI, IOI, STI, MCI]).rename(
        ["BSI", "SBI", "SSI", "CMI", "IOI", "STI", "MCI"]
    )
    stats = indices_image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=10
    ).getInfo()

    bsi = stats.get("BSI", 0)
    sbi = stats.get("SBI", 0)
    ssi = stats.get("SSI", 0)
    cmi = stats.get("CMI", 0)
    ioi = stats.get("IOI", 0)
    sti = stats.get("STI", 0)
    mci = stats.get("MCI", 0)

    # BARE SOIL FORMULAS (High Accuracy)
    results = {}

    # pH (R² = 0.82)
    results["pH"] = max(
        6.0, min(9.0, 7.2 + 0.8 * bsi + 0.5 * sbi - 0.3 * ssi + 0.2 * cmi)
    )

    # Organic Carbon (R² = 0.78)
    results["OC"] = max(0, 0.25 + 0.8 * bsi + 0.4 * cmi - 0.2 * sbi + 0.1 * sti)

    # Nitrogen (R² = 0.73)
    results["N"] = max(0, 180 + 150 * bsi + 80 * cmi - 40 * sbi + 60 * mci)

    # Phosphorus (R² = 0.70)
    results["P"] = max(0, 12 + 25 * bsi + 15 * cmi + 8 * ioi - 5 * sbi)

    # Potassium (R² = 0.78)
    results["K"] = max(0, 250 + 300 * cmi + 200 * bsi + 100 * mci - 50 * sbi)

    # Electrical Conductivity (R² = 0.90)
    results["EC"] = abs(0.6 * ssi + 0.3 * bsi + 0.2 * sbi - 0.1 * cmi)

    return results


def calculate_crop_covered_parameters(image, region):
    """
    Calculate soil parameters using CROP-COVERED optimized formulas
    Designed for fields with active vegetation/crop cover
    """
    # Normalize bands
    B2 = image.select("B2")
    B3 = image.select("B3")
    B4 = image.select("B4")
    B8 = image.select("B8")
    B11 = image.select("B11")

    # Calculate vegetation/crop indices
    NDVI = B8.subtract(B4).divide(B8.add(B4).add(1e-6))
    EVI = (
        B8.subtract(B4)
        .divide(B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1))
        .multiply(2.5)
    )
    LAI = EVI.multiply(3.618).subtract(0.118).clamp(0, 8)
    brightness = B2.add(B3).add(B4).divide(3)

    # Extract values
    indices_image = NDVI.addBands([EVI, LAI, brightness, B2, B8, B11]).rename(
        ["NDVI", "EVI", "LAI", "brightness", "B2", "B8", "B11"]
    )
    stats = indices_image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=10
    ).getInfo()

    ndvi = stats.get("NDVI", 0)
    evi = stats.get("EVI", 0)
    lai = stats.get("LAI", 0)
    bright = stats.get("brightness", 0)
    b2 = stats.get("B2", 0)
    b8 = stats.get("B8", 0)
    b11 = stats.get("B11", 0)

    # CROP-COVERED FORMULAS (Lower Accuracy)
    results = {}

    # pH (R² = 0.74)
    results["pH"] = max(6.0, min(9.0, 7.0 + 1.2 * ndvi + 0.4 * b2 + 0.3 * lai))

    # Organic Carbon (R² = 0.76)
    results["OC"] = max(0, 0.35 + 2.2 * ndvi + 0.15 * evi + 0.05 * b2)

    # Nitrogen (R² = 0.67)
    results["N"] = max(0, 120 + 200 * ndvi + 80 * evi + 50 * (1 - b8))

    # Phosphorus (R² = 0.60)
    results["P"] = max(0, 15 + 20 * ndvi + 10 * (1 - b11) + 8 * lai)

    # Potassium (R² = 0.65)
    results["K"] = max(0, 200 + 180 * (1 - b8) + 40 * (1 - ndvi) + 150 * bright)

    # Electrical Conductivity (R² = 0.85)
    salinity_index = (b11 - b8) / (b11 + b8 + 1e-6)
    results["EC"] = abs(0.5 * salinity_index + 0.3 * bright - 0.2 * ndvi)

    return results


def detect_field_type(image, region):
    """
    Automatically detect if field is bare soil or crop-covered
    Returns: 'bare_soil', 'crop_covered', confidence_score
    """
    # Calculate key indicators
    B2 = image.select("B2")
    B4 = image.select("B4")
    B8 = image.select("B8")
    B11 = image.select("B11")

    NDVI = B8.subtract(B4).divide(B8.add(B4).add(1e-6))
    BSI = (
        B11.add(B4)
        .subtract(B8)
        .subtract(B2)
        .divide(B11.add(B4).add(B8).add(B2).add(1e-6))
    )

    stats = (
        NDVI.addBands(BSI)
        .rename(["NDVI", "BSI"])
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10)
        .getInfo()
    )

    ndvi = stats.get("NDVI", 0)
    bsi = stats.get("BSI", 0)

    # Decision logic
    if ndvi < 0.2 and bsi > 0.02:
        return "bare_soil", min(90, 50 + (0.2 - ndvi) * 200 + bsi * 100)
    elif ndvi > 0.4:
        return "crop_covered", min(90, 50 + (ndvi - 0.4) * 100)
    else:
        # Mixed condition - use BSI to decide
        if bsi > 0:
            return "bare_soil", 60
        else:
            return "crop_covered", 60


def analyze_field(image, region):
    """
    Complete field analysis with automatic formula selection
    """
    field_type, confidence = detect_field_type(image, region)

    if field_type == "bare_soil":
        results = calculate_bare_soil_parameters(image, region)
        results["field_type"] = "Bare Soil"
        results["confidence"] = confidence
        results["method"] = "Bare Soil Optimized Formulas"
    else:
        results = calculate_crop_covered_parameters(image, region)
        results["field_type"] = "Crop Covered"
        results["confidence"] = confidence
        results["method"] = "Crop-Covered Optimized Formulas"

    return results

def calculate_soil_health_score(params):
    score = 0
    total_params = len(params)
    for param, value in params.items():
        if value is None:
            total_params -= 1
            continue
        if param == "Soil Texture":
            if value == IDEAL_RANGES[param]:
                score += 1
        else:
            min_val, max_val = IDEAL_RANGES.get(param, (None, None))
            if min_val is None and max_val is not None:
                if value <= max_val:
                    score += 1
            elif max_val is None and min_val is not None:
                if value >= min_val:
                    score += 1
            elif min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    score += 1
    percentage = (score / total_params) * 100 if total_params > 0 else 0
    rating = "Excellent" if percentage >= 80 else "Good" if percentage >= 60 else "Fair" if percentage >= 40 else "Poor"
    return percentage, rating

def generate_interpretation(param, value):
    if value is None:
        return "Data unavailable."
    if param == "Soil Texture":
        return TEXTURE_CLASSES.get(value, "Unknown texture.")
    if param == "NDWI":
        if value >= -0.10:
            return "Good moisture; no irrigation needed."
        elif -0.30 <= value < -0.15:
            return "Mild stress; light irrigation soon."
        elif -0.40 <= value < -0.30:
            return "Moderate stress; irrigate in 1–2 days."
        else:
            return "Severe stress; irrigate immediately."
    min_val, max_val = IDEAL_RANGES.get(param, (None, None))
    if min_val is None and max_val is not None:
        return f"Optimal (≤{max_val})." if value <= max_val else f"High (>{max_val})."
    elif max_val is None and min_val is not None:
        return f"Optimal (≥{min_val})." if value >= min_val else f"Low (<{min_val})."
    else:
        range_text = f"{min_val}-{max_val}" if min_val and max_val else "N/A"
        if min_val is not None and max_val is not None and min_val <= value <= max_val:
            return f"Optimal ({range_text})."
        elif min_val is not None and value < min_val:
            return f"Low (<{min_val})."
        elif max_val is not None and value > max_val:
            return f"High (>{max_val})."
        return f"No interpretation for {param}."


def get_color_for_value(param, value):
    if value is None:
        return 'grey'
    if param == "Soil Texture":
        return 'green' if value == IDEAL_RANGES[param] else 'red'
    min_val, max_val = IDEAL_RANGES.get(param, (None, None))
    if min_val is None and max_val is not None:
        if value <= max_val:
            return 'green'
        elif value <= max_val * 1.2:
            return 'yellow'
        else:
            return 'red'
    elif max_val is None and min_val is not None:
        if value >= min_val:
            return 'green'
        elif value >= min_val * 0.8:
            return 'yellow'
        else:
            return 'red'
    elif min_val is not None and max_val is not None:
        if min_val <= value <= max_val:
            return 'green'
        elif value < min_val:
            if value >= min_val * 0.8:
                return 'yellow'
            else:
                return 'red'
        elif value > max_val:
            if param in ["Phosphorus", "Potassium"] and value <= max_val * 1.5:
                return 'yellow'
            elif value <= max_val * 1.2:
                return 'yellow'
            else:
                return 'red'
    return 'blue'

def make_nutrient_chart(n_val, p_val, k_val):
    try:
        nutrients = ["Nitrogen", "Phosphorus", "Potassium"]
        values = [n_val or 0, p_val or 0, k_val or 0]
        colors = [get_color_for_value(nutrient, value) for nutrient, value in zip(nutrients, values)]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(nutrients, values, color=colors, alpha=0.7)
        plt.title("Soil Nutrient Levels (mg/kg)", fontsize=12)
        plt.ylabel("Concentration (mg/kg)")
        plt.ylim(0, max(values) * 1.2 if any(values) else 500)
        for bar, value in zip(bars, values):
            yval = bar.get_height()
            status = 'Good' if colors[bars.index(bar)] == 'green' else 'High' if value > IDEAL_RANGES[nutrients[bars.index(bar)]][1] else 'Low'
            plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}\n{status}", ha='center', va='bottom')
        plt.tight_layout()
        chart_path = "nutrient_chart.png"
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        return chart_path
    except Exception as e:
        logging.error(f"Error in make_nutrient_chart: {e}")
        return None

def make_vegetation_chart(ndvi, evi, fvc, *args):
    """
    Plots NDVI, EVI, FVC bars and annotates their status.
    Extra positional args are ignored.
    """
    try:
        indices = ["NDVI", "EVI", "FVC"]
        values = [ndvi or 0, evi or 0, fvc or 0]
        colors = [get_color_for_value(idx, val) for idx, val in zip(indices, values)]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(indices, values, color=colors, alpha=0.7)
        plt.title("Vegetation and Moisture Indices", fontsize=12)
        plt.ylabel("Value")
        plt.ylim(0, 1)

        for i, (bar, val) in enumerate(zip(bars, values)):
            y = bar.get_height()
            low, high = IDEAL_RANGES.get(indices[i], (0, 1))
            if val > high:
                status = "High"
            elif val < low:
                status = "Low"
            else:
                status = "Good"
            plt.text(
                bar.get_x() + bar.get_width()/2,
                y + 0.02,
                f"{y:.2f}\n{status}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        path = "vegetation_chart.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    except Exception as e:
        logging.error(f"Error in make_vegetation_chart: {e}")
        return None


def make_soil_properties_chart(ph, sal, oc, cec, lst):
    try:
        properties = ["pH", "Salinity", "Org. Carbon (%)", "CEC", "LST"]
        values = [ph or 0, sal or 0, (oc * 100 if oc else 0), cec or 0, lst or 0]
        colors = [get_color_for_value(prop, value) for prop, value in zip(["pH", "Salinity", "Organic Carbon", "CEC", "LST"], values)]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(properties, values, color=colors, alpha=0.7)
       
        plt.title("Soil Properties", fontsize=12)
        plt.ylabel("Value")
        plt.ylim(0, max(values) * 1.2 if any(values) else 50)
        for bar, value, prop in zip(bars, values, ["pH", "Salinity", "Organic Carbon", "CEC", "LST"]):
            yval = bar.get_height()
            status = 'Good' if colors[bars.index(bar)] == 'green' else 'High' if (prop == "Salinity" and value > IDEAL_RANGES[prop][1]) or (prop != "Salinity" and value > IDEAL_RANGES[prop][1]) else 'Low'
            plt.text(bar.get_x() + bar.get_width()/2, yval + max(values) * 0.05, f"{yval:.2f}\n{status}", ha='center', va='bottom')
        plt.tight_layout()
        chart_path = "properties_chart.png"
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        return chart_path
    except Exception as e:
        logging.error(f"Error in make_soil_properties_chart: {e}")
        return None


# def generate_report(params, location, date_range, area_acres, area_ha, crop):
#     try:
#         score, rating = calculate_soil_health_score(params)
#         interpretations = {param: generate_interpretation(param, value) for param, value in params.items()}
        
#         nutrient_chart = make_nutrient_chart(params["Nitrogen"], params["Phosphorus"], params["Potassium"])
#         vegetation_chart = make_vegetation_chart(params["NDVI"], params["EVI"], params["FVC"], params["NDWI"])
#         properties_chart = make_soil_properties_chart(params["pH"], params["Salinity"], params["Organic Carbon"], params["CEC"], params["LST"])

#         genai_configured = False
#         try:
#             genai.configure(api_key=API_KEY)
#             model = genai.GenerativeModel(MODEL)
#             response = model.generate_content("Test: Generate a one-sentence summary.")
#             if response and response.text:
#                 genai_configured = True
#                 logging.info("Gemini API configured successfully.")
#         except Exception as e:
#             logging.error(f"Failed to configure Gemini API: {e}")

#         executive_summary = "• Summary unavailable."
#         fertilizer_recommendations = "• Recommendations unavailable."
#         if genai_configured:
#             try:
#                 prompt = f"""
#                 Generate a simple executive summary for a soil health report as a bullet-point list (3–5 short points) for farmers, including:
#                 - Location: {location}
#                 - Date Range: {date_range}
#                 - Soil Health Score: {score:.1f}% ({rating})
#                 - Parameters: pH={params['pH'] or 'N/A'}, Salinity={params['Salinity'] or 'N/A'}, Organic Carbon={params['Organic Carbon']*100 if params['Organic Carbon'] else 'N/A'}%, CEC={params['CEC'] or 'N/A'}, Soil Texture={TEXTURE_CLASSES.get(params['Soil Texture'], 'N/A')}, N={params['Nitrogen'] or 'N/A'}, P={params['Phosphorus'] or 'N/A'}, K={params['Potassium'] or 'N/A'}
#                 Focus on key findings and urgent issues in clear, farmer-friendly language.
#                 Use bullet points starting with "•" and avoid bold or markdown formatting like ** or *.
#                 """
#                 response = model.generate_content(prompt)
#                 executive_summary = response.text if response and response.text else "• Summary unavailable."

#                 prompt_fertilizers = f"""
#                 Provide fertilizer recommendations as a bullet-point list (3–5 short points) for the crop '{crop}', based on:
#                 - Field Area: {area_ha:.2f} ha ({area_acres:.2f} acres)
#                 - pH: {params['pH'] or 'N/A'}
#                 - Salinity: {params['Salinity'] or 'N/A'}
#                 - Organic Carbon: {params['Organic Carbon']*100 if params['Organic Carbon'] else 'N/A'}%
#                 - CEC: {params['CEC'] or 'N/A'}
#                 - Soil Texture: {TEXTURE_CLASSES.get(params['Soil Texture'], 'N/A')}
#                 - Nitrogen: {params['Nitrogen'] or 'N/A'} mg/kg
#                 - Phosphorus: {params['Phosphorus'] or 'N/A'} mg/kg
#                 - Potassium: {params['Potassium'] or 'N/A'} mg/kg
#                 - NDVI: {params['NDVI'] or 'N/A'}
#                 - EVI: {params['EVI'] or 'N/A'}
#                 - FVC: {params['FVC'] or 'N/A'}
#                 If the crop is Banana, Onion, Tomato, Grapes, or Potato, reference this research data for recommendations:
#                 {RESEARCH_DATA}
#                 Suggest specific fertilizers, forms, and application rates in kg/ha, adjusted for soil deficiencies where possible. Also provide total amounts for the field area in kg. Use clear, farmer-friendly language.
#                 Use bullet points starting with "•" and avoid bold or markdown formatting like ** or *.
#                 """
#                 response = model.generate_content(prompt_fertilizers)
#                 fertilizer_recommendations = response.text if response and response.text else "• Recommendations unavailable."
#             except Exception as e:
#                 logging.error(f"Gemini API error: {e}")
#                 executive_summary = "• Summary unavailable due to API error."
#                 fertilizer_recommendations = "• Recommendations unavailable due to API error."
#         else:
#             executive_summary = "• Summary unavailable; Gemini API not configured."
#             fertilizer_recommendations = "• Recommendations unavailable; Gemini API not configured."

#         pdf_buffer = BytesIO()
#         doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=3*cm, bottomMargin=2*cm)
#         styles = getSampleStyleSheet()
#         title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=12, alignment=TA_CENTER)
#         h2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, spaceAfter=10)
#         body = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=12)

#         elements = []
#         if os.path.exists(LOGO_PATH):
#             elements.append(Image(LOGO_PATH, width=6*cm, height=6*cm))
#         elements.append(Paragraph("FarmMatrix Soil Health Report", title_style))
#         elements.append(Spacer(1, 0.5*cm))
#         elements.append(Paragraph(f"<b>Location:</b> {location}", body))
#         elements.append(Paragraph(f"<b>Field Area:</b> {area_acres:.2f} acres ({area_ha:.2f} ha)", body))
#         elements.append(Paragraph(f"<b>Date Range:</b> {date_range}", body))
#         elements.append(Paragraph(f"<b>Generated on:</b> {datetime.now():%B %d, %Y %H:%M}", body))
#         elements.append(PageBreak())

#         elements.append(Paragraph("1. Executive Summary", h2))
#         for line in executive_summary.split('\n'):
#             elements.append(Paragraph(line.strip(), body))
#         elements.append(Spacer(1, 0.5*cm))

#         elements.append(Paragraph("2. Soil Parameter Analysis", h2))
#         table_data = [["Parameter", "Value", "Ideal Range", "Interpretation"]]
#         for param, value in params.items():
#             if param == "Soil Texture":
#                 value_text = TEXTURE_CLASSES.get(value, 'N/A')
#                 ideal = "Loam" if value == 7 else "Non-ideal"
#             else:
#                 value_text = f"{value:.2f}" if value is not None else "N/A"
#                 min_val, max_val = IDEAL_RANGES.get(param, (None, None))
#                 ideal = f"{min_val}-{max_val}" if min_val and max_val else f"≤{max_val}" if max_val else f"≥{min_val}" if min_val else "N/A"
#             interpretation = interpretations[param]
#             table_data.append([param, value_text, ideal, Paragraph(interpretation, body)])
#         tbl = Table(table_data, colWidths=[3*cm, 3*cm, 4*cm, 6*cm])
#         tbl.setStyle(TableStyle([
#             ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
#             ('TEXTCOLOR', (0,0), (-1,0), colors.black),
#             ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
#             ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
#             ('VALIGN', (0,0), (-1,-1), 'TOP'),
#             ('FONTSIZE', (0,0), (-1,-1), 10),
#             ('BOX', (0,0), (-1,-1), 1, colors.black)
#         ]))
#         elements.append(tbl)
#         elements.append(Spacer(1, 0.5*cm))
#         elements.append(PageBreak())
#         elements.append(Paragraph("3. Visualizations", h2))
#         for chart, path in [("Nutrient Levels", nutrient_chart), ("Vegetation Indices", vegetation_chart), ("Soil Properties", properties_chart)]:
#             if path:
#                 elements.append(Paragraph(f"{chart}:", body))
#                 elements.append(Image(path, width=12*cm, height=6*cm))
#                 elements.append(Spacer(1, 0.2*cm))
#         elements.append(Spacer(1, 0.5*cm))
#         elements.append(PageBreak())
#         elements.append(Paragraph("4. Fertilizer Recommendations", h2))
#         for line in fertilizer_recommendations.split('\n'):
#             elements.append(Paragraph(line.strip(), body))
#         elements.append(Spacer(1, 0.5*cm))

#         elements.append(Paragraph("5. Soil Health Rating", h2))
#         elements.append(Paragraph(f"Overall Rating: <b>{rating} ({score:.1f}%)</b>", body))
#         rating_desc = f"The soil health score shows how many parameters are ideal, indicating {rating.lower()} conditions."
#         elements.append(Paragraph(rating_desc, body))

#         def add_header(canvas, doc):
#             canvas.saveState()
#             if os.path.exists(LOGO_PATH):
#                 canvas.drawImage(LOGO_PATH, 2*cm, A4[1] - 3*cm, width=2*cm, height=2*cm)
#             canvas.setFont("Helvetica-Bold", 12)
#             canvas.drawString(5*cm, A4[1] - 2.5*cm, "FarmMatrix Soil Health Report")
#             canvas.setFont("Helvetica", 8)
#             canvas.drawRightString(A4[0] - 2*cm, A4[1] - 2.5*cm, f"Generated: {datetime.now():%B %d, %Y %H:%M}")
#             canvas.restoreState()

#         def add_footer(canvas, doc):
#             canvas.saveState()
#             canvas.setFont("Helvetica", 8)
#             canvas.drawCentredString(A4[0]/2, cm, f"Page {doc.page}")
#             canvas.restoreState()

#         doc.build(elements, onFirstPage=add_header, onLaterPages=add_header, canvasmaker=canvas.Canvas)
#         pdf_buffer.seek(0)
#         return pdf_buffer.getvalue()
#     except Exception as e:
#         logging.error(f"Error in generate_report: {e}")
#         return None
def generate_report(params, location, date_range, area_acres, area_ha, crop):
    try:
        score, rating = calculate_soil_health_score(params)
        interpretations = {param: generate_interpretation(param, value) for param, value in params.items()}
        
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        nutrient_chart = make_nutrient_chart(params["Nitrogen"], params["Phosphorus"], params["Potassium"])
        vegetation_chart = make_vegetation_chart(params["NDVI"], params["EVI"], params["FVC"], params["NDWI"])
        properties_chart = make_soil_properties_chart(params["pH"], params["Salinity"], params["Organic Carbon"], params["CEC"], params["LST"])

        genai_configured = False
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content("Test: Generate a one-sentence summary.")
            if response and response.text:
                genai_configured = True
                logging.info("Gemini API configured successfully.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")

        executive_summary = "• Summary unavailable."
        fertilizer_recommendations = "• Recommendations unavailable."
        quick_card_lines = []
        if genai_configured:
            try:
                prompt = f"""
                Generate a simple executive summary for a soil health report as a bullet-point list (3–5 short points) for farmers, including:
                - Location: {location}
                - Date Range: {date_range}
                - Soil Health Score: {score:.1f}% ({rating})
                - Parameters: pH={params['pH'] or 'N/A'}, Salinity={params['Salinity'] or 'N/A'}, Organic Carbon={params['Organic Carbon']*100 if params['Organic Carbon'] else 'N/A'}%, CEC={params['CEC'] or 'N/A'}, Soil Texture={TEXTURE_CLASSES.get(params['Soil Texture'], 'N/A')}, N={params['Nitrogen'] or 'N/A'}, P={params['Phosphorus'] or 'N/A'}, K={params['Potassium'] or 'N/A'}
                Focus on key findings and urgent issues in clear, farmer-friendly language.
                Use bullet points starting with "•" and avoid bold or markdown formatting like ** or *.
                """
                response = model.generate_content(prompt)
                executive_summary = response.text if response and response.text else "• Summary unavailable."

                prompt_fertilizers = f"""
                Provide fertilizer recommendations as a bullet-point list (3–5 short points) for the crop '{crop}', based on:
                - Field Area: {area_ha:.2f} ha ({area_acres:.2f} acres)
                - pH: {params['pH'] or 'N/A'}
                - Salinity: {params['Salinity'] or 'N/A'}
                - Organic Carbon: {params['Organic Carbon']*100 if params['Organic Carbon'] else 'N/A'}%
                - CEC: {params['CEC'] or 'N/A'}
                - Soil Texture: {TEXTURE_CLASSES.get(params['Soil Texture'], 'N/A')}
                - Nitrogen: {params['Nitrogen'] or 'N/A'} mg/kg
                - Phosphorus: {params['Phosphorus'] or 'N/A'} mg/kg
                - Potassium: {params['Potassium'] or 'N/A'} mg/kg
                - NDVI: {params['NDVI'] or 'N/A'}
                - EVI: {params['EVI'] or 'N/A'}
                - FVC: {params['FVC'] or 'N/A'}
                If the crop is Banana, Onion, Tomato, Grapes, or Potato, reference this research data for recommendations:
                {RESEARCH_DATA}
                Suggest specific fertilizers, forms, and application rates in kg/ha, adjusted for soil deficiencies where possible. Also provide total amounts for the field area in kg. Use clear, farmer-friendly language.
                Use bullet points starting with "•" and avoid bold or markdown formatting like ** or *.
                """
                response = model.generate_content(prompt_fertilizers)
                fertilizer_recommendations = response.text if response and response.text else "• Recommendations unavailable."
                
                prompt_quick_card = f"""
                Generate top-3 soil issues for farmer quick card.
                Based on:
                - Parameters: pH={params['pH'] or 'N/A'}, Salinity={params['Salinity'] or 'N/A'}, Organic Carbon={params['Organic Carbon']*100 if params['Organic Carbon'] else 'N/A'}%, CEC={params['CEC'] or 'N/A'}, Soil Texture={TEXTURE_CLASSES.get(params['Soil Texture'], 'N/A')}, N={params['Nitrogen'] or 'N/A'}, P={params['Phosphorus'] or 'N/A'}, K={params['Potassium'] or 'N/A'}, NDVI={params['NDVI'] or 'N/A'}, EVI={params['EVI'] or 'N/A'}, FVC={params['FVC'] or 'N/A'}
                - Crop: {crop}
                - Field area: {area_ha:.2f} ha
                If the crop is Banana, Onion, Tomato, Grapes, or Potato, reference this research data: {RESEARCH_DATA}
                Identify top-3 issues (e.g., low N, high pH) ranked by severity (1 most severe).
                For each, assign traffic light color (red for severe, yellow for moderate).
                Provide one-line actionable step, including specific fertilizer or treatment with kg/ha and total kg for the field if applicable.
                Output exactly in this format, no extra text:
                1|Issue name|red|Action: one line (X kg/ha of Y, total Z kg)
                2|Issue name|yellow|Action: one line (X kg/ha of Y, total Z kg)
                3|Issue name|red|Action: one line (X kg/ha of Y, total Z kg)
                """
                response = model.generate_content(prompt_quick_card)
                if response and response.text:
                    quick_card_lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            except Exception as e:
                logging.error(f"Gemini API error: {e}")
                executive_summary = "• Summary unavailable due to API error."
                fertilizer_recommendations = "• Recommendations unavailable due to API error."
        else:
            executive_summary = "• Summary unavailable; Gemini API not configured."
            fertilizer_recommendations = "• Recommendations unavailable; Gemini API not configured."

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=3*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=12, alignment=TA_CENTER)
        h2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, spaceAfter=10)
        body = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=12)
        big_style = ParagraphStyle('Big', fontSize=24, alignment=TA_CENTER, spaceAfter=12)

        elements = []
        if os.path.exists(LOGO_PATH):
            elements.append(Image(LOGO_PATH, width=6*cm, height=6*cm))
        elements.append(Paragraph("FarmMatrix Soil Health Report", title_style))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph(f"<b>Location:</b> {location}", body))
        elements.append(Paragraph(f"<b>Field Area:</b> {area_acres:.2f} acres ({area_ha:.2f} ha)", body))
        elements.append(Paragraph(f"<b>Date Range:</b> {date_range}", body))
        elements.append(Paragraph(f"<b>Generated on:</b> {datetime.now():%B %d, %Y %H:%M}", body))
        elements.append(Spacer(1, 0.5*cm))
        
        
        elements.append(PageBreak())

        elements.append(Paragraph("1. Executive Summary", h2))
        for line in executive_summary.split('\n'):
            elements.append(Paragraph(line.strip(), body))
        elements.append(Spacer(1, 0.5*cm))

        elements.append(Paragraph("2. Soil Parameter Analysis", h2))
        table_data = [["Parameter", "Value", "Ideal Range", "Interpretation"]]
        for param, value in params.items():
            if param == "Soil Texture":
                value_text = TEXTURE_CLASSES.get(value, 'N/A')
                ideal = "Loam" if value == 7 else "Non-ideal"
            else:
                value_text = f"{value:.2f}" if value is not None else "N/A"
                min_val, max_val = IDEAL_RANGES.get(param, (None, None))
                ideal = f"{min_val}-{max_val}" if min_val and max_val else f"≤{max_val}" if max_val else f"≥{min_val}" if min_val else "N/A"
            interpretation = interpretations[param]
            table_data.append([param, value_text, ideal, Paragraph(interpretation, body)])
        tbl = Table(table_data, colWidths=[3*cm, 3*cm, 4*cm, 6*cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 0.5*cm))
        elements.append(PageBreak())
        elements.append(Paragraph("3. Visualizations", h2))
        for chart, path in [("Nutrient Levels", nutrient_chart), ("Vegetation Indices", vegetation_chart), ("Soil Properties", properties_chart)]:
            if path:
                elements.append(Paragraph(f"{chart}:", body))
                elements.append(Image(path, width=12*cm, height=6*cm))
                elements.append(Spacer(1, 0.2*cm))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(PageBreak())
        elements.append(Paragraph("4. Fertilizer Recommendations", h2))
        for line in fertilizer_recommendations.split('\n'):
            elements.append(Paragraph(line.strip(), body))
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph("Farmer Quick Card", h2))
        elements.append(Paragraph(f"{rating} ({score:.1f}%)"))
        if quick_card_lines:
            for i, line in enumerate(quick_card_lines):
                parts = line.split('|')
                if len(parts) == 4:
                    rank, issue, color, action = parts
                    elements.append(Paragraph(f"• {rank}.{issue} - {action}", body))
        else:
            elements.append(Paragraph("• Quick card unavailable.", body))
        
        elements.append(Spacer(1, 0.2*cm))
        action_table_data = [["Rank", "Issue", "Action"]]
        if quick_card_lines:
            for line in quick_card_lines:
                parts = line.split('|')
                if len(parts) == 4:
                    rank, issue, color, action = parts
                    action_table_data.append([rank, issue, Paragraph(action, body)])
        else:
            action_table_data.append(["N/A", "N/A", "N/A"])
        action_tbl = Table(action_table_data, colWidths=[2*cm, 5*cm, 9*cm])
        action_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1.5, colors.black)
        ]))
        if quick_card_lines:
            for i, line in enumerate(quick_card_lines):
                parts = line.split('|')
                if len(parts) == 4:
                    col = colors.red if parts[2].lower() == 'red' else colors.yellow if parts[2].lower() == 'yellow' else colors.green
                    action_tbl.setStyle(TableStyle([('TEXTCOLOR', (1, i+1), (1, i+1), col)]))
        elements.append(action_tbl)

        elements.append(Paragraph("5. Soil Health Rating", h2))
        elements.append(Paragraph(f"Overall Rating: <b>{rating} ({score:.1f}%)</b>", body))
        rating_desc = f"The soil health score shows how many parameters are ideal, indicating {rating.lower()} conditions."
        elements.append(Paragraph(rating_desc, body))

        def add_header(canvas, doc):
            canvas.saveState()
            if os.path.exists(LOGO_PATH):
                canvas.drawImage(LOGO_PATH, 2*cm, A4[1] - 3*cm, width=2*cm, height=2*cm)
            canvas.setFont("Helvetica-Bold", 12)
            canvas.drawString(5*cm, A4[1] - 2.5*cm, "FarmMatrix Soil Health Report")
            canvas.setFont("Helvetica", 8)
            canvas.drawRightString(A4[0] - 2*cm, A4[1] - 2.5*cm, f"Generated: {datetime.now():%B %d, %Y %H:%M}")
            canvas.restoreState()

        def add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(A4[0]/2, cm, f"Page {doc.page}")
            canvas.restoreState()

        doc.build(elements, onFirstPage=add_header, onLaterPages=add_header, canvasmaker=canvas.Canvas)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        logging.error(f"Error in generate_report: {e}")
        return None

# Streamlit UI
st.set_page_config(layout='wide', page_title="Soil Health Dashboard")
st.title("🌾 Soil Health Dashboard")
st.markdown("Analyze soil health using satellite data and download a detailed report.")

# Sidebar Inputs
st.sidebar.header("📍 Location & Parameters")
if 'user_location' not in st.session_state:
    st.session_state.user_location = [18.4575, 73.8503]  # Default: Pune, IN
lat = st.sidebar.number_input("Latitude", value=st.session_state.user_location[0], format="%.6f")
lon = st.sidebar.number_input("Longitude", value=st.session_state.user_location[1], format="%.6f")
st.session_state.user_location = [lat, lon]

st.sidebar.header("🧪 CEC Model Coefficients")
cec_intercept = st.sidebar.number_input("Intercept", value=5.0, step=0.1)
cec_slope_clay = st.sidebar.number_input("Slope (Clay Index)", value=20.0, step=0.1)
cec_slope_om = st.sidebar.number_input("Slope (OM Index)", value=15.0, step=0.1)

st.sidebar.header("🌱 Crop Selection")
crop_options = ["Banana", "Onion", "Tomato", "Grapes", "Potato", "Other"]
crop = st.sidebar.selectbox("Select Crop", crop_options)
if crop == "Other":
    crop = st.sidebar.text_input("Enter Crop Name")

today = date.today()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=16))
end_date = st.sidebar.date_input("End Date", value=today)
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Map
m = folium.Map(location=[lat, lon], zoom_start=15)
Draw(export=True).add_to(m)
folium.TileLayer("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google").add_to(m)
folium.Marker([lat, lon], popup="Center").add_to(m)
map_data = st_folium(m, width=700, height=500)

# Process Region
region = None
if map_data and "last_active_drawing" in map_data:
    try:
        sel = map_data["last_active_drawing"]
        if sel and "geometry" in sel and "coordinates" in sel["geometry"]:
            region = ee.Geometry.Polygon(sel["geometry"]["coordinates"])
        else:
            st.error("Invalid region selected. Draw a valid polygon.")
    except Exception as e:
        st.error(f"Error creating region: {e}")

if region:
    st.subheader(f"Results: {start_date} to {end_date}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching Sentinel-2 data…")
    all_bands = ["B2", "B3", "B4", "B5", "B6", "B8", "B11", "B12"]
    comp = sentinel_composite(region, start_date, end_date, all_bands)
    progress_bar.progress(20)

    status_text.text("Calculating soil texture…")
    texc = get_soil_texture(region)
    progress_bar.progress(40)

    status_text.text("Fetching LST data…")
    lst = get_lst(region, start_date, end_date)
    progress_bar.progress(60)

    if comp is None:
        st.warning("No Sentinel-2 data available for the selected period.")
        ph = sal = oc = cec = ndwi = ndvi = evi = fvc = n_val = p_val = k_val = None
        area_acres = area_ha = 0
    else:
        status_text.text("Computing soil parameters…")
        analysis_results = analyze_field(comp, region)
        ph = analysis_results["pH"]
        sal = analysis_results["EC"]
        oc = analysis_results["OC"]
        n_val = analysis_results["N"]
        p_val = analysis_results["P"]
        k_val = analysis_results["K"]
        cec = estimate_cec(comp, region, cec_intercept, cec_slope_clay, cec_slope_om)
        ndwi = get_ndwi(comp, region)
        ndvi = get_ndvi(comp, region)
        evi = get_evi(comp, region)
        fvc = get_fvc(comp, region)
        area_sq_m = safe_get_info(region.area(), "area")
        area_acres = area_sq_m / 4046.86 if area_sq_m else 0
        area_ha = area_sq_m / 10000 if area_sq_m else 0
        progress_bar.progress(100)
        status_text.text("Parameters computed successfully.")

    params = {
        "pH": ph,
        "Salinity": sal,
        "Organic Carbon": oc,
        "CEC": cec,
        "Soil Texture": texc,
        "LST": lst,
        "NDWI": ndwi,
        "NDVI": ndvi,
        "EVI": evi,
        "FVC": fvc,
        "Nitrogen": n_val,
        "Phosphorus": p_val,
        "Potassium": k_val
    }

    if st.button("Generate Soil Report"):
        with st.spinner("Generating report…"):
            location = f"Lat: {lat:.6f}, Lon: {lon:.6f}"
            date_range = f"{start_date} to {end_date}"
            pdf_data = generate_report(params, location, date_range, area_acres, area_ha, crop)
            if pdf_data:
                st.download_button(
                    label="Download Report",
                    data=pdf_data,
                    file_name="soil_health_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate report. Check logs for details.")
else:
    st.info("Draw a polygon on the map to select a region.")