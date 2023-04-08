import numpy as np
import pandas as pd
import tensorflow as tf

# Import Altair data as CSV file
channel = pd.read_csv('channel_data.csv')

# Split data into features and labels
features = channel[['UAV x coordinate', 'UAV y coordinate', 'UAV z coordinate', 'received power', 'packet latency']].values
labels = channel[['RSSI', 'Throughput']].values

# Define the DQNs model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, input_dim=5, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(2)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(features, labels, epochs=500, batch_size=32)

# Test the model
test_features = np.array([[1.0, 2.0, 3.0, -20.0, 50.0]])
predictions = model.predict(test_features)
print(predictions)
