import streamlit as st
import numpy as np
import requests
from supabase import create_client, Client

# Placeholder for Supabase and API keys - replace with your actual keys or use st.secrets
SUPABASE_URL = "https://wxlhbigzassvjribednu.supabase.co"  # e.g., "https://yourproject.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind4bGhiaWd6YXNzdmpyaWJlZG51Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg2OTcwMzMsImV4cCI6MjA3NDI3MzAzM30.qRuPSTbAZDoWf5Xi9S_Gfj4sDgU_rKGsntfC5ZP42Bc"
LOCATIONIQ_KEY = "pk.f4fe45d86d54ab31f7c6b69f04637b84"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to compute polygon centroid using numpy (no shapely needed)
def compute_centroid(points):
    points = np.array(points)
    if not np.all(points[0] == points[-1]):
        points = np.vstack((points, points[0]))  # Close the polygon
    x = points[:, 0]  # lon
    y = points[:, 1]  # lat
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if A == 0:
        raise ValueError("Polygon has zero area")
    cx = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    cy = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    return cx, cy  # lon, lat

# Streamlit App
st.title("FarmMatrix Community Feature")

# User Input for Name and Phone
name = st.text_input("Enter your name")
phone = st.text_input("Enter your phone number")

if name and phone:
    # Check or create user
    user_data = supabase.table("users").select("user_id").eq("phone", phone).execute()
    if user_data.data:
        user_id = user_data.data[0]["user_id"]
        st.write(f"Welcome back, {name}! User ID: {user_id}")
    else:
        insert_user = supabase.table("users").insert({"name": name, "phone": phone}).execute()
        user_id = insert_user.data[0]["user_id"]
        st.write(f"New user created! User ID: {user_id}")

    # Input for Polygon Coordinates
    coords_input = st.text_area(
        "Enter polygon coordinates as 'lon1 lat1, lon2 lat2, ...' (repeat first point at end to close)",
        "73.939972 20.145934, 73.940738 20.145805, 73.940615 20.144991, 73.939856 20.145073, 73.939972 20.145934"
    )

    if st.button("Submit Polygon and Get Location"):
        try:
            # Parse coordinates (assuming 'lon lat, lon lat, ...')
            pairs = [pair.strip().split() for pair in coords_input.split(",")]
            points = [(float(lon), float(lat)) for lon, lat in pairs]

            # Compute centroid
            lon, lat = compute_centroid(points)
            st.write(f"Centroid: Longitude {lon}, Latitude {lat}")

            # Insert or update field
            polygon_wkt = f"POLYGON(({', '.join([f'{p[0]} {p[1]}' for p in points])}))"
            field_insert = supabase.table("fields").insert({
                "user_id": user_id,
                "polygon_coordinates": polygon_wkt
            }).execute()

            # Call LocationIQ API
            api_url = "https://us1.locationiq.com/v1/reverse"
            params = {"key": LOCATIONIQ_KEY, "lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
            response = requests.get(api_url, params=params)
            data = response.json()

            if "address" in data:
                village = data["address"].get("village", "Unknown")
                taluka = data["address"].get("county", "Unknown")  # Often taluka
                district = data["address"].get("state_district", "Unknown")
                st.write(f"Village: {village}, Taluka: {taluka}, District: {district}")

                # Update user_locations
                supabase.table("user_locations").upsert({
                    "user_id": user_id,
                    "village": village,
                    "taluka": taluka,
                    "district": district
                }).execute()

                # Store in session for display
                st.session_state["location"] = {"village": village, "taluka": taluka, "district": district}
            else:
                st.error("API response error")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Display and Join Groups if location is available
    if "location" in st.session_state:
        location = st.session_state["location"]
        group_types = [
            ("village", location["village"]),
            ("taluka", location["taluka"]),
            ("district", location["district"])
        ]

        for g_type, g_name in group_types:
            # Check if group exists, create if not
            group_query = supabase.table("groups").select("group_id").eq("group_type", g_type).eq("group_name", g_name).execute()
            if group_query.data:
                group_id = group_query.data[0]["group_id"]
            else:
                group_insert = supabase.table("groups").insert({
                    "group_type": g_type,
                    "group_name": g_name
                }).execute()
                group_id = group_insert.data[0]["group_id"]

            # Check if user is member
            member_query = supabase.table("group_members").select("user_id").eq("group_id", group_id).eq("user_id", user_id).execute()
            is_member = bool(member_query.data)

            st.subheader(f"{g_type.capitalize()} Group: {g_name}")
            if is_member:
                st.write("You are already a member.")
            else:
                if st.button(f"Join {g_type.capitalize()} Group"):
                    supabase.table("group_members").insert({
                        "group_id": group_id,
                        "user_id": user_id
                    }).execute()
                    st.success("Joined successfully!")
                    st.rerun()

            # Show members if member
            if is_member:
                members = supabase.table("group_members") \
                    .select("users(name, phone)") \
                    .eq("group_id", group_id) \
                    .join("users", "group_members.user_id == users.user_id") \
                    .execute()
                if members.data:
                    st.write("Group Members:")
                    for member in members.data:
                        st.write(f"- {member['users']['name']} ({member['users']['phone']})")
                else:
                    st.write("No other members yet.")