# The-Learning-Machines
change
from sklearn.cluster import KMeans
import numpy as np

# Example data
stations = np.array([[40.7128, -74.0060, 'station1'],
                     [37.7749, -122.4194, 'station2'],
                     [41.8781, -87.6298, 'station3'],
                     [34.0522, -118.2437, 'station4'],
                     [51.5074, -0.1278, 'station5']])

# Create a KMeans clustering model with k clusters
kmeans = KMeans(n_clusters=5)

# Fit the model to the station coordinates
kmeans.fit(stations[:,:2])

# Predict the cluster labels for the missing coordinates
missing_coords = np.array([[40.7142, -74.0064],
                           [41.8765, -87.6523],
                           [51.5099, -0.1180]])
missing_labels = kmeans.predict(missing_coords)

# Assign the closest station id to each missing coordinate
for i, label in enumerate(missing_labels):
    closest_station = stations[kmeans.labels_ == label][-1, 2]
    print(f"Missing station id {i+1} is closest to {closest_station}.")
