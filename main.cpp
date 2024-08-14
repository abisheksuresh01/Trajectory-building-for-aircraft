/*
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

// Define PI
const double PI = 3.141592653589793;

// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}

// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    // WGS84 ellipsoid constants
    double a = 6378137.0;  // Equatorial radius in meters
    double f = 1 / 298.257223563;  // Flattening
    double e_sq = f * (2 - f);  // Square of eccentricity

    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);

    double N = a / sqrt(1 - e_sq * pow(sin(lat_rad), 2));

    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);

    return {X, Y, Z};
}

// ECEF to ENU conversion
std::vector<double> ecef_to_enu(double X, double Y, double Z,
                                double X_ref, double Y_ref, double Z_ref,
                                double lat_ref, double lon_ref) {
    double lat_ref_rad = deg2rad(lat_ref);
    double lon_ref_rad = deg2rad(lon_ref);

    double dx = X - X_ref;
    double dy = Y - Y_ref;
    double dz = Z - Z_ref;

    double sin_lat = sin(lat_ref_rad);
    double cos_lat = cos(lat_ref_rad);
    double sin_lon = sin(lon_ref_rad);
    double cos_lon = cos(lon_ref_rad);

    double t[3] = {dx, dy, dz};

    double R[3][3] = {
        {-sin_lon, cos_lon, 0},
        {-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat},
        {cos_lat * cos_lon, cos_lat * sin_lon, sin_lat}
    };

    std::vector<double> enu(3);
    enu[0] = R[0][0] * t[0] + R[0][1] * t[1] + R[0][2] * t[2];
    enu[1] = R[1][0] * t[0] + R[1][1] * t[1] + R[1][2] * t[2];
    enu[2] = R[2][0] * t[0] + R[2][1] * t[1] + R[2][2] * t[2];

    return enu;
}

// Save data for plotting
void save_data_for_plotting(const std::vector<double>& start_ecef, const std::vector<double>& end_ecef, const std::string& filename) {
    std::ofstream dataFile(filename);
    dataFile << start_ecef[0] << " " << start_ecef[1] << " " << start_ecef[2] << std::endl;
    dataFile << end_ecef[0] << " " << end_ecef[1] << " " << end_ecef[2] << std::endl;
    dataFile.close();
}

int main() {
    // Example usage
    double start_latitude = 40.0;  // degrees
    double start_longitude = -74.0;  // degrees
    double start_altitude = 30.0;  // meters

    double end_latitude = 42.0;  // degrees
    double end_longitude = -75.0;  // degrees
    double end_altitude = 50.0;  // meters

    // Convert start and end locations to ECEF coordinates
    auto start_ecef = llh_to_ecef(start_latitude, start_longitude, start_altitude);
    auto end_ecef = llh_to_ecef(end_latitude, end_longitude, end_altitude);

    // Convert end ECEF to ENU coordinates at the start location
    auto end_enu = ecef_to_enu(end_ecef[0], end_ecef[1], end_ecef[2],
                               start_ecef[0], start_ecef[1], start_ecef[2],
                               start_latitude, start_longitude);

    // Output the results
    std::cout << "Start Location ECEF Coordinates: X = " << start_ecef[0]
              << ", Y = " << start_ecef[1] << ", Z = " << start_ecef[2] << std::endl;
    std::cout << "End Location ECEF Coordinates: X = " << end_ecef[0]
              << ", Y = " << end_ecef[1] << ", Z = " << end_ecef[2] << std::endl;
    std::cout << "End Location ENU Coordinates relative to Start: East = " << end_enu[0]
              << ", North = " << end_enu[1] << ", Up = " << end_enu[2] << std::endl;

    // Save data to file for plotting
    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(start_ecef, end_ecef, data_filename);

    // Call Python script to plot the trajectory
    std::string python_command = "python plot_trajectory.py";
    int result = system(python_command.c_str());
    if (result != 0) {
        std::cerr << "Error running the Python plotting script." << std::endl;
    }

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>

const double PI = 3.14159265358979323846;
const double a = 6378137.0; // WGS-84 semi-major axis
const double f = 1 / 298.257223563; // WGS-84 flattening

// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}

// Convert radians to degrees
double rad2deg(double radians) {
    return radians * 180.0 / PI;
}

// Haversine formula to calculate the great-circle distance between two points
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6371000; // Earth radius in meters
    double phi1 = deg2rad(lat1);
    double phi2 = deg2rad(lat2);
    double delta_phi = deg2rad(lat2 - lat1);
    double delta_lambda = deg2rad(lon2 - lon1);

    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) *
               sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return R * c;
}

// Calculate intermediate points on the geodesic
std::vector<std::vector<double>> calculate_geodesic_points(double lat1, double lon1, double lat2, double lon2, int num_points) {
    std::vector<std::vector<double>> points;
    points.push_back({lat1, lon1});

    for (int i = 1; i < num_points; i++) {
        double frac = (double)i / (double)num_points;
        double A = sin((1 - frac) * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double B = sin(frac * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);

        double x = A * cos(deg2rad(lat1)) * cos(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * cos(deg2rad(lon2));
        double y = A * cos(deg2rad(lat1)) * sin(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * sin(deg2rad(lon2));
        double z = A * sin(deg2rad(lat1)) + B * sin(deg2rad(lat2));

        double lat = rad2deg(atan2(z, sqrt(x * x + y * y)));
        double lon = rad2deg(atan2(y, x));

        points.push_back({lat, lon});
    }

    points.push_back({lat2, lon2});
    return points;
}

// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    double e_sq = f * (2 - f); // Square of eccentricity
    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);

    double N = a / sqrt(1 - e_sq * pow(sin(lat_rad), 2));

    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);

    return {X, Y, Z};
}

// ECEF to ENU conversion
std::vector<double> ecef_to_enu(double X, double Y, double Z,
                                double X_ref, double Y_ref, double Z_ref,
                                double lat_ref, double lon_ref) {
    double lat_ref_rad = deg2rad(lat_ref);
    double lon_ref_rad = deg2rad(lon_ref);

    double dx = X - X_ref;
    double dy = Y - Y_ref;
    double dz = Z - Z_ref;

    double sin_lat = sin(lat_ref_rad);
    double cos_lat = cos(lat_ref_rad);
    double sin_lon = sin(lon_ref_rad);
    double cos_lon = cos(lon_ref_rad);

    double t[3] = {dx, dy, dz};

    double R[3][3] = {
        {-sin_lon, cos_lon, 0},
        {-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat},
        {cos_lat * cos_lon, cos_lat * sin_lon, sin_lat}
    };

    std::vector<double> enu(3);
    enu[0] = R[0][0] * t[0] + R[0][1] * t[1] + R[0][2] * t[2];
    enu[1] = R[1][0] * t[0] + R[1][1] * t[1] + R[1][2] * t[2];
    enu[2] = R[2][0] * t[0] + R[2][1] * t[1] + R[2][2] * t[2];

    return enu;
}

// Save data for plotting
void save_data_for_plotting(const std::vector<std::vector<double>>& geodesic_points, const std::string& filename) {
    std::ofstream dataFile(filename);
    for (const auto& point : geodesic_points) {
        auto ecef = llh_to_ecef(point[0], point[1], 0.0);
        dataFile << ecef[0] << " " << ecef[1] << " " << ecef[2] << std::endl;
    }
    dataFile.close();
}

int main() {
    // Example usage
    double start_latitude = 40.0;  // degrees
    double start_longitude = -74.0;  // degrees
    double end_latitude = 42.0;  // degrees
    double end_longitude = -75.0;  // degrees
    int num_intermediate_points = 100;

    // Calculate geodesic points
    auto geodesic_points = calculate_geodesic_points(start_latitude, start_longitude, end_latitude, end_longitude, num_intermediate_points);

    // Save data to file for plotting
    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(geodesic_points, data_filename);

    // Call Python script to plot the trajectory
    std::string python_command = "python plot_trajectory.py";
    int result = system(python_command.c_str());
    if (result != 0) {
        std::cerr << "Error running the Python plotting script." << std::endl;
    }

    return 0;
}

*/

/*
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

const double PI = 3.14159265358979323846;
const double a = 6378137.0; // WGS-84 semi-major axis
const double f = 1 / 298.257223563; // WGS-84 flattening

// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}

// Convert radians to degrees
double rad2deg(double radians) {
    return radians * 180.0 / PI;
}

// Haversine formula to calculate the great-circle distance between two points
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6371000; // Earth radius in meters
    double phi1 = deg2rad(lat1);
    double phi2 = deg2rad(lat2);
    double delta_phi = deg2rad(lat2 - lat1);
    double delta_lambda = deg2rad(lon2 - lon1);

    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) *
               sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return R * c;
}

// Calculate intermediate points on the geodesic
std::vector<std::vector<double>> calculate_geodesic_points(double lat1, double lon1, double lat2, double lon2, int num_points) {
    std::vector<std::vector<double>> points;
    
    for (int i = 0; i <= num_points; i++) {
        double frac = (double)i / (double)num_points;
        double A = sin((1 - frac) * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double B = sin(frac * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);

        double x = A * cos(deg2rad(lat1)) * cos(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * cos(deg2rad(lon2));
        double y = A * cos(deg2rad(lat1)) * sin(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * sin(deg2rad(lon2));
        double z = A * sin(deg2rad(lat1)) + B * sin(deg2rad(lat2));

        double lat = rad2deg(atan2(z, sqrt(x * x + y * y)));
        double lon = rad2deg(atan2(y, x));

        points.push_back({lat, lon});
    }

    return points;
}

// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    double e_sq = f * (2 - f); // Square of eccentricity
    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);

    double N = a / sqrt(1 - e_sq * pow(sin(lat_rad), 2));

    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);

    return {X, Y, Z};
}

// Save data for plotting
void save_data_for_plotting(const std::vector<std::vector<double>>& points, const std::string& filename) {
    std::ofstream dataFile(filename);
    for (const auto& point : points) {
        auto ecef = llh_to_ecef(point[0], point[1], point[2]);
        dataFile << ecef[0] << " " << ecef[1] << " " << ecef[2] << std::endl;
    }
    dataFile.close();
}

int main() {
    // Flight parameters
    double start_latitude = 40.0;  // degrees
    double start_longitude = -74.0;  // degrees
    double end_latitude = 42.0;  // degrees
    double end_longitude = -75.0;  // degrees
    double cruise_altitude = 7000;  // meters (cruise altitude)
    int num_intermediate_points = 100; // Points on the cruise path

    // Phase 1: Ascent - Compute ascent path points
    std::vector<std::vector<double>> ascent_path;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * frac;
        ascent_path.push_back({start_latitude, start_longitude, altitude});
    }

    // Phase 2: Cruise - Compute cruise path points
    std::vector<std::vector<double>> cruise_path = calculate_geodesic_points(start_latitude, start_longitude, end_latitude, end_longitude, num_intermediate_points);
    for (auto& point : cruise_path) {
        point.push_back(cruise_altitude); // Set cruise altitude
    }

    // Phase 3: Descent - Compute descent path points
    std::vector<std::vector<double>> descent_path;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * (1 - frac);
        descent_path.push_back({end_latitude, end_longitude, altitude});
    }

    // Combine all phases
    std::vector<std::vector<double>> complete_path;
    complete_path.insert(complete_path.end(), ascent_path.begin(), ascent_path.end());
    complete_path.insert(complete_path.end(), cruise_path.begin(), cruise_path.end());
    complete_path.insert(complete_path.end(), descent_path.begin(), descent_path.end());

    // Save data to file for plotting
    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(complete_path, data_filename);

    // Call Python script to plot the trajectory
    std::string python_command = "python plot_trajectory.py";
    int result = system(python_command.c_str());
    if (result != 0) {
        std::cerr << "Error running the Python plotting script." << std::endl;
    }

    return 0;
}



#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

const double PI = 3.14159265358979323846;
const double a = 6378137.0; // WGS-84 semi-major axis
const double f = 1 / 298.257223563; // WGS-84 flattening
const double e_sq = f * (2.0 - f); // Square of eccentricity

// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}

// Convert radians to degrees
double rad2deg(double radians) {
    return radians * 180.0 / PI;
}

// Haversine formula to calculate the great-circle distance between two points
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6371000; // Earth radius in meters
    double phi1 = deg2rad(lat1);
    double phi2 = deg2rad(lat2);
    double delta_phi = deg2rad(lat2 - lat1);
    double delta_lambda = deg2rad(lon2 - lon1);

    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) *
               sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return R * c;
}

// Calculate intermediate points on the geodesic
std::vector<std::vector<double>> calculate_geodesic_points(double lat1, double lon1, double lat2, double lon2, int num_points) {
    std::vector<std::vector<double>> points;
    
    for (int i = 0; i <= num_points; i++) {
        double frac = (double)i / (double)num_points;
        double A = sin((1 - frac) * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double B = sin(frac * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);

        double x = A * cos(deg2rad(lat1)) * cos(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * cos(deg2rad(lon2));
        double y = A * cos(deg2rad(lat1)) * sin(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * sin(deg2rad(lon2));
        double z = A * sin(deg2rad(lat1)) + B * sin(deg2rad(lat2));

        double lat = rad2deg(atan2(z, sqrt(x * x + y * y)));
        double lon = rad2deg(atan2(y, x));

        points.push_back({lat, lon});
    }

    return points;
}

// Function to simulate a turn by adjusting the heading towards the target heading
double adjust_heading(double current_heading, double target_heading, double adjustment_rate) {
    double delta = fmod(target_heading - current_heading + 360.0, 360.0);
    if (delta > 180.0) {
        delta -= 360.0;
    }
    return fmod(current_heading + (delta > 0 ? adjustment_rate : -adjustment_rate) + 360.0, 360.0);
}

// ECEF to LLA conversion
std::vector<double> ecef_to_lla(double X, double Y, double Z) {
    double eps = 1e-9; // Convergence criterion
    double e = sqrt(e_sq);
    double b = a * sqrt(1 - e_sq);
    double p = sqrt(X * X + Y * Y);
    double th = atan2(a * Z, b * p);

    double lon = atan2(Y, X);
    double lat = atan2(Z + e_sq * b * pow(sin(th), 3), p - e_sq * a * pow(cos(th), 3));
    double N = a / sqrt(1 - e_sq * sin(lat) * sin(lat));
    double alt = p / cos(lat) - N;

    // Refine latitude and altitude with the iterative method
    double lat0 = lat;
    do {
        lat0 = lat;
        N = a / sqrt(1 - e_sq * sin(lat) * sin(lat));
        alt = p / cos(lat) - N;
        lat = atan2(Z + e_sq * N * sin(lat), p);
    } while (fabs(lat - lat0) > eps);

    return {rad2deg(lat), rad2deg(lon), alt};
}

// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);

    double N = a / sqrt(1 - e_sq * sin(lat_rad) * sin(lat_rad));

    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);

    return {X, Y, Z};
}

// Save data for plotting
void save_data_for_plotting(const std::vector<std::vector<double>>& points, const std::string& filename) {
    std::ofstream dataFile(filename);
    for (const auto& point : points) {
        dataFile << point[0] << " " << point[1] << " " << point[2] << std::endl;
    }
    dataFile.close();
}

int main() {
    double start_latitude = 40.0;
    double start_longitude = -74.0;
    double end_latitude = 42.0;
    double end_longitude = -75.0;
    double cruise_altitude = 7000; // meters
    int num_intermediate_points = 100;
    double start_runway_heading = 30.0; // degrees from North
    double end_runway_heading = 110.0; // degrees from North
    double heading_rate = 5.0;

    std::vector<std::vector<double>> complete_path;

    // Phase 1: Ascent and initial turn
    double current_heading = start_runway_heading;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * frac;
        double adjusted_heading = adjust_heading(current_heading, 90.0, heading_rate * frac);
        double delta_lat = frac * (end_latitude - start_latitude);
        double delta_lon = frac * (end_longitude - start_longitude);
        auto ecef_point = llh_to_ecef(start_latitude + delta_lat, start_longitude + delta_lon, altitude);
        auto lla_point = ecef_to_lla(ecef_point[0], ecef_point[1], ecef_point[2]);
        complete_path.push_back(lla_point);
    }

    // Phase 2: Cruise along the geodesic path
    auto cruise_path = calculate_geodesic_points(start_latitude, start_longitude, end_latitude, end_longitude, num_intermediate_points);
    for (auto& point : cruise_path) {
        auto ecef_point = llh_to_ecef(point[0], point[1], cruise_altitude);
        auto lla_point = ecef_to_lla(ecef_point[0], ecef_point[1], ecef_point[2]);
        complete_path.push_back(lla_point);
    }

    // Phase 3: Descent and final turn
    current_heading = end_runway_heading;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * (1 - frac);
        double adjusted_heading = adjust_heading(90.0, current_heading, heading_rate * (1 - frac));
        double delta_lat = frac * (end_latitude - start_latitude);
        double delta_lon = frac * (end_longitude - start_longitude);
        auto ecef_point = llh_to_ecef(end_latitude - delta_lat, end_longitude - delta_lon, altitude);
        auto lla_point = ecef_to_lla(ecef_point[0], ecef_point[1], ecef_point[2]);
        complete_path.push_back(lla_point);
    }

    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(complete_path, data_filename);

    std::string python_command = "python plot_trajectory.py";
    int result = system(python_command.c_str());
    if (result != 0) {
        std::cerr << "Error running the Python plotting script." << std::endl;
    }

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>


const double PI = 3.14159265358979323846;
const double a = 6378137.0; // WGS-84 semi-major axis
const double f = 1 / 298.257223563; // WGS-84 flattening


// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}


// Convert radians to degrees
double rad2deg(double radians) {
    return radians * 180.0 / PI;
}


// Haversine formula to calculate the great-circle distance between two points
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6371000; // Earth radius in meters
    double phi1 = deg2rad(lat1);
    double phi2 = deg2rad(lat2);
    double delta_phi = deg2rad(lat2 - lat1);
    double delta_lambda = deg2rad(lon2 - lon1);


    double a = sin(delta_phi / 2) * sin(delta_phi / 2) +
               cos(phi1) * cos(phi2) *
               sin(delta_lambda / 2) * sin(delta_lambda / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));


    return R * c;
}


// Calculate intermediate points on the geodesic
std::vector<std::vector<double>> calculate_geodesic_points(double lat1, double lon1, double lat2, double lon2, int num_points) {
    std::vector<std::vector<double>> points;
   
    for (int i = 0; i <= num_points; i++) {
        double frac = (double)i / (double)num_points;
        double A = sin((1 - frac) * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double B = sin(frac * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);


        double x = A * cos(deg2rad(lat1)) * cos(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * cos(deg2rad(lon2));
        double y = A * cos(deg2rad(lat1)) * sin(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * sin(deg2rad(lon2));
        double z = A * sin(deg2rad(lat1)) + B * sin(deg2rad(lat2));


        double lat = rad2deg(atan2(z, sqrt(x * x + y * y)));
        double lon = rad2deg(atan2(y, x));


        points.push_back({lat, lon});
    }


    return points;
}


// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    double e_sq = f * (2 - f); // Square of eccentricity
    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);


    double N = a / sqrt(1 - e_sq * pow(sin(lat_rad), 2));


    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);


    return {X, Y, Z};
}


// Save data for plotting
void save_data_for_plotting(const std::vector<std::vector<double>>& points, const std::string& filename) {
    std::ofstream dataFile(filename);
    for (const auto& point : points) {
        auto ecef = llh_to_ecef(point[0], point[1], point[2]);
        dataFile << ecef[0] << " " << ecef[1] << " " << ecef[2] << std::endl;
    }
    dataFile.close();
}


int main() {
    // Flight parameters
    double start_latitude = 40.0;  // degrees
    double start_longitude = -74.0;  // degrees
    double end_latitude = 42.0;  // degrees
    double end_longitude = -75.0;  // degrees
    double cruise_altitude = 7000;  // meters (cruise altitude)
    int num_intermediate_points = 100; // Points on the cruise path


    // Phase 1: Ascent - Compute ascent path points
    std::vector<std::vector<double>> ascent_path;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * frac;
        ascent_path.push_back({start_latitude, start_longitude, altitude});
    }


    // Phase 2: Cruise - Compute cruise path points
    std::vector<std::vector<double>> cruise_path = calculate_geodesic_points(start_latitude, start_longitude, end_latitude, end_longitude, num_intermediate_points);
    for (auto& point : cruise_path) {
        point.push_back(cruise_altitude); // Set cruise altitude
    }


    // Phase 3: Descent - Compute descent path points
    std::vector<std::vector<double>> descent_path;
    for (int i = 0; i <= num_intermediate_points; ++i) {
        double frac = (double)i / (double)num_intermediate_points;
        double altitude = cruise_altitude * (1 - frac);
        descent_path.push_back({end_latitude, end_longitude, altitude});
    }


    // Combine all phases
    std::vector<std::vector<double>> complete_path;
    complete_path.insert(complete_path.end(), ascent_path.begin(), ascent_path.end());
    complete_path.insert(complete_path.end(), cruise_path.begin(), cruise_path.end());
    complete_path.insert(complete_path.end(), descent_path.begin(), descent_path.end());


    // Save data to file for plotting
    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(complete_path, data_filename);


    // Call Python script to plot the trajectory
    std::string python_command = "python plot_trajectory.py";
    int result = system(python_command.c_str());
    if (result != 0) {
        std::cerr << "Error running the Python plotting script." << std::endl;
    }


    return 0;
}



#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// Constants for the Earth's radius
const double R = 6371.0; // Kilometers

// Converts degrees to radians
double toRadians(const double degree) {
    return (degree * M_PI / 180.0);
}

// Function to calculate waypoints using great-circle distance
std::vector<std::pair<double, double>> computeWaypoints(double lat1, double lon1, double lat2, double lon2, int num_points = 100) {
    std::vector<std::pair<double, double>> waypoints;
    double dLon = toRadians(lon2 - lon1);
    lat1 = toRadians(lat1);
    lat2 = toRadians(lat2);
    lon1 = toRadians(lon1);

    for (int i = 0; i <= num_points; i++) {
        double f = double(i) / num_points;
        double A = sin((1 - f) * dLon) / sin(dLon);
        double B = sin(f * dLon) / sin(dLon);
        double x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon1 + dLon);
        double y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon1 + dLon);
        double z = A * sin(lat1) + B * sin(lat2);
        double lat = atan2(z, sqrt(x * x + y * y));
        double lon = atan2(y, x);
        waypoints.push_back({lat * 180.0 / M_PI, lon * 180.0 / M_PI});
    }
    return waypoints;
}

int main() {
    // Starting and ending coordinates
    double startLat = 34.0522, startLon = -118.2437; // Los Angeles
    double endLat = 40.7128, endLon = -74.0060;      // New York

    auto waypoints = computeWaypoints(startLat, startLon, endLat, endLon);

    // Writing waypoints to a CSV file
    std::ofstream outFile("trajectory.csv");
    outFile << "Latitude,Longitude\n";
    for (const auto& point : waypoints) {
        outFile << point.first << "," << point.second << "\n";
    }
    outFile.close();

    return 0;
}

*/

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

// Define constants
const double PI = 3.14159265358979323846;
const double a = 6378137.0; // WGS-84 semi-major axis
const double f = 1 / 298.257223563; // WGS-84 flattening

// Convert degrees to radians
double deg2rad(double degrees) {
    return degrees * PI / 180.0;
}

// Convert radians to degrees
double rad2deg(double radians) {
    return radians * 180.0 / PI;
}

// LLA to ECEF conversion
std::vector<double> llh_to_ecef(double latitude, double longitude, double altitude) {
    double e_sq = f * (2 - f); // Square of eccentricity
    double lat_rad = deg2rad(latitude);
    double lon_rad = deg2rad(longitude);
    double N = a / sqrt(1 - e_sq * pow(sin(lat_rad), 2));
    double X = (N + altitude) * cos(lat_rad) * cos(lon_rad);
    double Y = (N + altitude) * cos(lat_rad) * sin(lon_rad);
    double Z = ((1 - e_sq) * N + altitude) * sin(lat_rad);
    return {X, Y, Z};
}

// Calculate intermediate points on the geodesic
std::vector<std::vector<double>> calculate_geodesic_points(double lat1, double lon1, double lat2, double lon2, int num_points) {
    std::vector<std::vector<double>> points;
    for (int i = 0; i <= num_points; i++) {
        double frac = (double)i / (double)num_points;
        double A = sin((1 - frac) * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double B = sin(frac * haversine_distance(lat1, lon1, lat2, lon2) / a) / sin(haversine_distance(lat1, lon1, lat2, lon2) / a);
        double x = A * cos(deg2rad(lat1)) * cos(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * cos(deg2rad(lon2));
        double y = A * cos(deg2rad(lat1)) * sin(deg2rad(lon1)) + B * cos(deg2rad(lat2)) * sin(deg2rad(lon2));
        double z = A * sin(deg2rad(lat1)) + B * sin(deg2rad(lat2));
        double lat = rad2deg(atan2(z, sqrt(x * x + y * y)));
        double lon = rad2deg(atan2(y, x));
        points.push_back({lat, lon});
    }
    return points;
}

// Save data for plotting
void save_data_for_plotting(const std::vector<std::vector<double>> &points, const std::string &filename) {
    std::ofstream dataFile(filename);
    for (const auto &point : points) {
        auto ecef = llh_to_ecef(point[0], point[1], point[2]);
        dataFile << ecef[0] << " " << ecef[1] << " " << ecef[2] << std::endl;
    }
    dataFile.close();
}

int main() {
    // Flight parameters
    double start_latitude = 34.0522; // Los Angeles
    double start_longitude = -118.2437; // Los Angeles
    double end_latitude = 40.7128; // New York
    double end_longitude = -74.0060; // New York
    double cruise_altitude = 10000; // meters (cruise altitude)
    int num_intermediate_points = 100; // Points on the cruise path

    // Compute the geodesic path
    auto path_points = calculate_geodesic_points(start_latitude, start_longitude, end_latitude, end_longitude, num_intermediate_points);
    for (auto &point : path_points) {
        point.push_back(cruise_altitude); // Set cruise altitude
    }

    // Save data to file for plotting
    std::string data_filename = "trajectory_data.txt";
    save_data_for_plotting(path_points, data_filename);

    return 0;
}
