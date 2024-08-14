import math
import numpy as np

def llh_to_ecef(latitude, longitude, altitude):
    """
    Convert Latitude (degrees), Longitude (degrees), Altitude (meters)
    to ECEF coordinates (X, Y, Z in meters).
    """
    # Earth's radius and flattening factor from WGS84
    a = 6378137.0  # Equatorial radius in meters
    f = 1 / 298.257223563  # Flattening
    e_sq = f * (2 - f)  # Square of eccentricity

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Prime vertical radius of curvature
    N = a / math.sqrt(1 - e_sq * math.sin(lat_rad) ** 2)

    # Calculate ECEF coordinates
    X = (N + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - e_sq) * N + altitude) * math.sin(lat_rad)

    return np.array([X, Y, Z])

def ecef_to_enu(X, Y, Z, X_ref, Y_ref, Z_ref, lat_ref, lon_ref):
    """
    Convert ECEF coordinates (X, Y, Z) to ENU coordinates at reference point (X_ref, Y_ref, Z_ref)
    with reference latitude and longitude (lat_ref, lon_ref).
    """
    # Convert reference latitude and longitude from degrees to radians
    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)

    # Translation vector from reference point to the point of interest
    t = np.array([X - X_ref, Y - Y_ref, Z - Z_ref])

    # Transformation matrix from ECEF to ENU coordinates
    R = np.array([
        [-math.sin(lon_ref_rad), math.cos(lon_ref_rad), 0],
        [-math.sin(lat_ref_rad)*math.cos(lon_ref_rad), -math.sin(lat_ref_rad)*math.sin(lon_ref_rad), math.cos(lat_ref_rad)],
        [math.cos(lat_ref_rad)*math.cos(lon_ref_rad), math.cos(lat_ref_rad)*math.sin(lon_ref_rad), math.sin(lat_ref_rad)]
    ])

    # ENU coordinates
    enu = R.dot(t)
    return enu

def convert_start_end_to_ecef_enu(start_lat, start_lon, start_alt, end_lat, end_lon, end_alt):
    """
    Convert start and end locations from latitude, longitude, altitude to ECEF coordinates and then to ENU coordinates.
    """
    # Convert start location to ECEF
    start_ecef = llh_to_ecef(start_lat, start_lon, start_alt)
    
    # Convert end location to ECEF
    end_ecef = llh_to_ecef(end_lat, end_lon, end_alt)
    
    # Convert end ECEF to ENU coordinates at the start location
    enu = ecef_to_enu(end_ecef[0], end_ecef[1], end_ecef[2],
                      start_ecef[0], start_ecef[1], start_ecef[2],
                      start_lat, start_lon)
    
    return start_ecef, end_ecef, enu

# Example usage:
start_latitude = 40.0   # degrees
start_longitude = -74.0 # degrees
start_altitude = 30.0   # meters

end_latitude = 42.0   # degrees
end_longitude = -75.0 # degrees
end_altitude = 50.0   # meters

start_ecef, end_ecef, end_enu = convert_start_end_to_ecef_enu(start_latitude, start_longitude, start_altitude,
                                                             end_latitude, end_longitude, end_altitude)

print(f"Start Location ECEF Coordinates: X = {start_ecef[0]}, Y = {start_ecef[1]}, Z = {start_ecef[2]}")
print(f"End Location ECEF Coordinates: X = {end_ecef[0]}, Y = {end_ecef[1]}, Z = {end_ecef[2]}")
print(f"End Location ENU Coordinates relative to Start: East = {end_enu[0]}, North = {end_enu[1]}, Up = {end_enu[2]}")
