import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #type:ignore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv("unsupervised_anomaly_detection.csv")

    # Handle missing values if any
    df.fillna(df.median(), inplace=True)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data, df

# Step 2: Visualize data distribution
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title("Feature Distribution")
    plt.show()

# Step 3: Train the Isolation Forest model
def train_isolation_forest(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)

    # Predict anomalies (-1: anomaly, 1: normal)
    predictions = model.predict(data)
    return model, predictions

# Step 4: Evaluate the model
def evaluate_model(predictions, ground_truth):
    # Convert predictions and ground_truth to binary labels (1: normal, 0: anomaly)
    binary_predictions = np.where(predictions == 1, 1, 0)
    binary_truth = np.where(ground_truth == 1, 1, 0)

    precision = precision_score(binary_truth, binary_predictions)
    recall = recall_score(binary_truth, binary_predictions)
    f1 = f1_score(binary_truth, binary_predictions)

    conf_matrix = confusion_matrix(binary_truth, binary_predictions)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return conf_matrix

# Step 5: Visualize results
def plot_results(data, predictions):
    plt.figure(figsize=(10, 6))
    
    # Mark anomalies and normal points
    anomalies = data[predictions == -1]
    normal = data[predictions == 1]

    plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal', alpha=0.6)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly', alpha=0.6)

    plt.title("Anomaly Detection Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Main function
def main():
    file_path = "unsupervised_anomaly_detection.csv"  # Replace with your file path

    print("Unsupervised Anomaly Detection Data Imported")
    # Load and preprocess data
    scaled_data, original_data = load_and_preprocess_data(file_path)

    # Visualize data distribution
    visualize_data(original_data)

    # Train the model
    model, predictions = train_isolation_forest(scaled_data)

    # Assume ground_truth column exists for evaluation (replace this if ground truth is unavailable)
    if 'ground_truth' in original_data.columns:
        ground_truth = original_data['ground_truth']
        conf_matrix = evaluate_model(predictions, ground_truth)

        # Plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # Visualize the results
    plot_results(scaled_data, predictions)

if __name__ == "__main__":
    main()