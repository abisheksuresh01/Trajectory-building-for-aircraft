# Aircraft Trajectory Simulation and Visualization

The codes simulate and visualize the trajectory of an aircraft using Earth-Centered, Earth-Fixed (ECEF) coordinates and provide both 2D and 3D visualizations of the flight path.

## Files Description

### `coordinate.py`

This Python script includes functions to convert latitude, longitude, and altitude to ECEF coordinates, and ECEF coordinates to East-North-Up (ENU) coordinates. It serves as a fundamental module for transforming geographic coordinates for further processing in trajectory calculations.

### `main.cpp`

A C++ program that integrates various components of the aircraft trajectory simulation. This script potentially serves as the main entry point for processing trajectory data, incorporating calculations and logic defined in other scripts.

### `plot_trajectory.py`

A Python script that reads trajectory data from `trajectory_data.txt` and plots the trajectory in both 2D and 3D perspectives using Matplotlib and mpl_toolkits for 3D plotting. This script highlights different segments of the flight path, such as start, mid-cruise, and end.

### `trajectory_data.txt`

Contains simulated trajectory data in ECEF coordinates. This data is used by `plot_trajectory.py` to visualize the flight path of the aircraft.

### `trajectory.csv`

A CSV file that might be used for detailed analysis or additional data representation purposes not covered in the provided scripts.

### Executables

- `trajectory.exe`
- `trajectory_calculator.exe`

These executable files are compiled from C++ source codes (not provided in detail here) and are likely used for performing intensive computations or simulations that are part of the trajectory analysis.

## How to Run

Ensure Python is installed and has the required libraries (`numpy`, `matplotlib`) available. For C++ files, ensure your environment is set up to compile and run C++ programs.

To plot the trajectory, run:
```bash
python plot_trajectory.py
