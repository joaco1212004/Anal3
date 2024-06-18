#%%
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = np.load('test_predictions.npy', allow_pickle=True)
i_data = data['i']
o_data = data['o']
p_data = data['p']

# Extract predicted labels
y_pred = np.argmax(p_data, axis=-1).flatten()
y_true = o_data.flatten()

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate classification report
class_report = classification_report(y_true, y_pred, output_dict=True)

# Print metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Optionally, print input data and their shapes
print("Input data shape:", i_data.shape)
print("Output data shape:", o_data.shape)
print("Predicted data shape:", p_data.shape)
print("Input data sample:", i_data[:5])
print("Output data sample:", o_data[:5])
print("Predicted data sample:", p_data[:5])

# %%