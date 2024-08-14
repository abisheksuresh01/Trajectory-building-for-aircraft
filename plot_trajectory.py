"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Plotting ECEF coordinates in 3D
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[0], Y[0], Z[0], color='blue', label='Start (ECEF)', s=100)
ax.scatter(X[1], Y[1], Z[1], color='red', label='End (ECEF)', s=100)
ax.plot(X, Y, Z, color='black', linewidth=2)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('ECEF Coordinates')
ax.legend()

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Plotting ECEF coordinates in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(X, Y, Z, color='black', linewidth=2, label='Trajectory')

ax.scatter(X[0], Y[0], Z[0], color='blue', label='Start (ECEF)', s=100)
ax.scatter(X[-1], Y[-1], Z[-1], color='red', label='End (ECEF)', s=100)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('ECEF Coordinates of Trajectory')
ax.legend()

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Plotting ECEF coordinates in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(X, Y, Z, color='black', linewidth=2, label='Flight Path')

# Highlight the start, cruise, and end points
ax.scatter(X[0], Y[0], Z[0], color='blue', label='Start (ECEF)', s=100)
ax.scatter(X[-1], Y[-1], Z[-1], color='red', label='End (ECEF)', s=100)
ax.scatter(X[len(X)//2], Y[len(Y)//2], Z[len(Z)//2], color='green', label='Mid-Cruise (ECEF)', s=100)

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory of the Aircraft')
ax.legend()

plt.show()

"""
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Plotting ECEF coordinates in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(X, Y, Z, color='black', linewidth=2, label='Flight Path')

# Highlight the start, cruise, and end points
ax.scatter(X[0], Y[0], Z[0], color='blue', label='Start (ECEF)', s=100)
ax.scatter(X[-1], Y[-1], Z[-1], color='red', label='End (ECEF)', s=100)
ax.scatter(X[len(X)//2], Y[len(Y)//2], Z[len(Z)//2], color='green', label='Mid-Cruise (ECEF)', s=100)

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory of the Aircraft')
ax.legend()

plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
lat, lon, alt = data[:, 0], data[:, 1], data[:, 2]

# Convert altitude from meters to kilometers for better readability
alt /= 1000

# Plotting latitude and longitude on a 2D map
plt.figure(figsize=(12, 8))

# Use scatter plot for altitude color coding
plt.scatter(lon, lat, c=alt, cmap='viridis', marker='o')
plt.colorbar(label='Altitude (km)')
plt.grid(True)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Flight Path on Earth')

# Enhance the plot with start and end markers
plt.scatter(lon[0], lat[0], color='red', label='Start', zorder=5, s=100)
plt.scatter(lon[-1], lat[-1], color='blue', label='End', zorder=5, s=100)
plt.legend()

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('trajectory_data.txt')
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]  # ECEF coordinates

# Assuming constant speed and equal spacing in time between data points
aircraft_speed = 900  # km/h
time_interval = 1  # assuming each data point is spaced 1 minute apart

# Calculate distances between consecutive points
distances = np.sqrt(np.diff(X)**2 + np.diff(Y)**2 + np.diff(Z)**2)
cumulative_distances = np.cumsum(distances)
cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Start from zero

# Convert distances from meters to kilometers
cumulative_distances /= 1000

# Time data (in hours), assuming each point represents 1 minute interval
times = np.arange(len(Z)) / 60

# Plot Range vs Altitude
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(cumulative_distances, Z, label='Range vs Altitude')
plt.xlabel('Range (km)')
plt.ylabel('Altitude (m)')
plt.title('Range vs Altitude')
plt.grid(True)
plt.legend()

# Plot Time vs Altitude
plt.subplot(1, 3, 2)
plt.plot(times, Z, label='Time vs Altitude', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('Altitude (m)')
plt.title('Time vs Altitude')
plt.grid(True)
plt.legend()

# Plot Time vs Range
plt.subplot(1, 3, 3)
plt.plot(times, cumulative_distances, label='Time vs Range', color='green')
plt.xlabel('Time (hours)')
plt.ylabel('Range (km)')
plt.title('Time vs Range')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

# Read CSV file
df = pd.read_csv('trajectory.csv')

# Set up the map
fig, ax = plt.subplots()
map = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=50,
              llcrnrlon=-130, urcrnrlon=-60, resolution='i', ax=ax)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral', lake_color='aqua')

# Convert latitude and longitude to map projection coordinates
x, y = map(df['Longitude'].values, df['Latitude'].values)

# Plot trajectory
map.plot(x, y, marker=None, color='m')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Simulating data
time = np.linspace(0, 10, 100)  # Time from 0 to 10 hours
altitude = np.linspace(1000, 35000, 100)  # Altitude from 1000 ft to 35000 ft
range_km = time**2 * 30  # Range in km, assuming some acceleration

# Plot 1: Range vs Altitude
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(range_km, altitude, '-b')
plt.xlabel('Range (km)')
plt.ylabel('Altitude (ft)')
plt.title('Range vs Altitude')
plt.grid(True)

# Plot 2: Time vs Altitude
plt.subplot(1, 3, 2)
plt.plot(time, altitude, '-r')
plt.xlabel('Time (hours)')
plt.ylabel('Altitude (ft)')
plt.title('Time vs Altitude')
plt.grid(True)

# Plot 3: Time vs Range
plt.subplot(1, 3, 3)
plt.plot(time, range_km, '-g')
plt.xlabel('Time (hours)')
plt.ylabel('Range (km)')
plt.title('Time vs Range')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('trajectory_data.txt')

# Separate the coordinates
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Plotting ECEF coordinates in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(X, Y, Z, color='black', linewidth=2, label='Flight Path')

# Highlight the start, cruise, and end points
ax.scatter(X[0], Y[0], Z[0], color='blue', label='Start (ECEF)', s=100)
ax.scatter(X[-1], Y[-1], Z[-1], color='red', label='End (ECEF)', s=100)
ax.scatter(X[len(X)//2], Y[len(Y)//2], Z[len(Z)//2], color='green', label='Mid-Cruise (ECEF)', s=100)

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory of the Aircraft')
ax.legend()

plt.show()
