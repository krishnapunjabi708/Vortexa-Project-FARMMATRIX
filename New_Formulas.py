import ee
import streamlit as st
import datetime
import pandas as pd


# The provided functions
def calculate_bare_soil_parameters(image, region):
    """
    Calculate soil parameters using BARE SOIL optimized formulas
    Designed for bare/harvested fields with minimal vegetation
    """
    # Normalize bands
    B2 = image.select("B2").divide(10000)
    B3 = image.select("B3").divide(10000)
    B4 = image.select("B4").divide(10000)
    B5 = image.select("B5").divide(10000)
    B6 = image.select("B6").divide(10000)
    B8 = image.select("B8").divide(10000)
    B11 = image.select("B11").divide(10000)
    B12 = image.select("B12").divide(10000)

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
    B2 = image.select("B2").divide(10000)
    B3 = image.select("B3").divide(10000)
    B4 = image.select("B4").divide(10000)
    B8 = image.select("B8").divide(10000)
    B11 = image.select("B11").divide(10000)

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
    B2 = image.select("B2").divide(10000)
    B4 = image.select("B4").divide(10000)
    B8 = image.select("B8").divide(10000)
    B11 = image.select("B11").divide(10000)

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


# Streamlit app
st.title("Farmer Soil Reports for Fursungi Fields")

st.write(
    "Generating soil reports for the 8 predefined fields in Fursungi. Sample collection date: August 6, 2025."
)

# Earth Engine Authentication
if "ee_initialized" not in st.session_state:
    try:
        ee.Initialize()
        st.session_state["ee_initialized"] = True
    except ee.EEException:
        auth_code = st.text_input(
            "Authenticate Earth Engine: Visit https://code.earthengine.google.com/client-auth and enter the verification code here:"
        )
        if auth_code:
            ee.Authenticate(auth_code=auth_code)
            ee.Initialize()
            st.session_state["ee_initialized"] = True
            st.success("Authenticated successfully!")
    except Exception as e:
        st.error(f"Error initializing Earth Engine: {e}")

# Predefined polygons (corrected coordinates)
polygons = [
    [
        [74.003821, 18.473055],
        [74.004304, 18.472959],
        [74.004222, 18.472675],
        [74.003748, 18.472748],
    ],
    [
        [74.003636, 18.472719],
        [74.003051, 18.472860],
        [74.003118, 18.473110],
        [74.003704, 18.472992],
    ],
    [
        [74.003127, 18.465447],
        [74.003312, 18.465338],
        [74.003251, 18.465168],
        [74.003031, 18.465246],
    ],
    [
        [74.001177, 18.470410],
        [74.001528, 18.470647],
        [74.002475, 18.470511],
        [74.002422, 18.470282],
    ],
    [
        [74.001440, 18.469539],
        [74.001587, 18.469907],
        [74.001959, 18.469890],
        [74.001874, 18.469521],
    ],
    [
        [74.003689, 18.475094],
        [74.003389, 18.475055],
        [74.003709, 18.474216],
        [74.003560, 18.474266],
    ],
    [
        [74.001779, 18.475162],
        [74.001730, 18.474971],
        [74.001242, 18.475027],
        [74.001284, 18.475233],
    ],
    [
        [73.983860, 18.473279],
        [73.983261, 18.473406],
        [73.983308, 18.473985],
        [73.983862, 18.473947],
    ],
    [
        [73.890226, 18.654585],
        [73.890100, 18.654623],
        [73.890032, 18.653696],
        [73.900157, 18.653691],
    ],
    [
        [73.890059, 18.654614],
        [73.889931, 18.654624],
        [73.889875, 18.653941],
        [73.889987, 18.653957],
    ],
    [
        [73.890074, 18.646245],
        [73.890043, 18.646479],
        [73.889308, 18.646495],
        [73.889318, 18.646219],
    ],
]

# Date range around sample collection (30 days before August 6, 2025)
ee_start = ee.Date("2025-07-07")
ee_end = ee.Date("2025-08-07")

# Button to generate reports
if st.button("Generate All Soil Reports"):
    for i, coords in enumerate(polygons):
        region = ee.Geometry.Polygon(coords)

        # Fetch latest Sentinel-2 image in the date range
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee_start, ee_end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        image_info = collection.first().getInfo()
        if image_info:
            image = ee.Image(image_info["id"]).clip(region)

            # Analyze
            results = analyze_field(image, region)

            st.subheader(f"Soil Report for Field {i + 1} (Fursungi)")
            st.write("Sample Collection Date: August 6, 2025")
            st.write(f"Field Type: {results['field_type']}")
            st.write(f"Confidence: {results['confidence']}%")
            st.write(f"Method: {results['method']}")
            st.write("Parameters:")
            params = {
                k: v
                for k, v in results.items()
                if k not in ["field_type", "confidence", "method"]
            }
            st.table(params)

            # Prepare CSV for download
            df = pd.DataFrame([params])
            df.insert(0, "Field", f"Field {i + 1}")
            df.insert(1, "Location", "Fursungi")
            df.insert(2, "Sample Date", "August 6, 2025")
            df.insert(3, "Field Type", results["field_type"])
            df.insert(4, "Confidence", f"{results['confidence']}%")
            df.insert(5, "Method", results["method"])
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"Download Report for Field {i + 1} as CSV",
                data=csv,
                file_name=f"soil_report_field_{i + 1}.csv",
                mime="text/csv",
            )
        else:
            st.error(
                f"No suitable Sentinel-2 image found for Field {i + 1} between 2025-07-07 and 2025-08-07. Try adjusting the date range."
            )
