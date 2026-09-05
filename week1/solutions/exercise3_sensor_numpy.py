"""Exercise 3: Sensor Data Analysis with NumPy — solution."""

import numpy as np
import matplotlib.pyplot as plt

# Raw data (hourly temperatures over 5 days in °C)
day1 = [12.1, 12.5, 13.0, 14.2, 15.1, 16.0, 15.8, 15.3]
day2 = [11.9, 12.2, 12.8, 14.0, 15.2, 16.1, 15.9, 15.2]
day3 = [10.8, 11.5, 12.3, 13.1, 14.9, 15.8, 15.5, 14.9]
day4 = [9.7, 10.1, 11.0, 12.2, 13.8, 14.7, 14.3, 13.5]
day5 = [8.9, 9.5, 10.3, 11.7, 12.9, 13.8, 13.2, 12.6]

# Task 1: Data Conversion
temps_c = np.array([day1, day2, day3, day4, day5])  # Shape (5, 8)
temps_f = temps_c * 9 / 5 + 32

# Task 2: Statistical Analysis
daily_means = np.mean(temps_c, axis=1)
daily_ranges = np.max(temps_c, axis=1) - np.min(temps_c, axis=1)
hottest_hour = np.argmax(np.mean(temps_c, axis=0))

print("temps_c shape:", temps_c.shape)
print("Daily means (°C):", daily_means)
print("Daily ranges (°C):", daily_ranges)
print("Hottest hour index:", hottest_hour)

# Task 3: Data Filtering
masked_temps = np.ma.masked_where(temps_c <= 15, temps_c)
high_temp_counts = np.sum(temps_c > 14, axis=1)

print("Masked temps (> 15°C):\n", masked_temps)
print("Counts > 14°C per day:", high_temp_counts)

# Task 4: Visualisation (Bonus)
plt.plot(temps_c.T)  # Transpose for hourly view
plt.xlabel("Hour of Day")
plt.ylabel("Temperature (°C)")
plt.title("Daily Temperature Trends")
plt.legend(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])
plt.show()
