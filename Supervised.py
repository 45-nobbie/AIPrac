import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Category', 'Message']].copy()
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    return df

# Visualize the dataset
def visualize_data(df):
    plt.figure(figsize=(8, 4))
    df['Category'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
    plt.title('Category Distribution')
    plt.xlabel('Category (0=Ham, 1=Spam)')
    plt.ylabel('Count')
    plt.show()

# Preprocess text and split the dataset
def preprocess_and_split(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['Message']).toarray()
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix

# Visualize confusion matrix and accuracy
def plot_results(conf_matrix, accuracy):
    plt.figure(figsize=(14, 6))

    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Plot accuracy bar graph
    plt.subplot(1, 2, 2)
    plt.bar(["Testing Accuracy"], [accuracy], color="orange")
    plt.ylim(0.8, 1.0)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

# Main function
def main():
    file_path = 'SPAM.csv'  # Replace with your file path
    df = load_data(file_path)
    visualize_data(df)

    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    model = train_model(X_train, y_train)

    accuracy, conf_matrix = evaluate_model(model, X_test, y_test)
    plot_results(conf_matrix, accuracy)

# Run the program
if __name__ == "__main__":
    main()