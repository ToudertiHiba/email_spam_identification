# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load Data
def load_data(file_path):
    # Read data from a CSV file into a DataFrame
    df = pd.read_csv(file_path)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Convert text data into TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

# Step 3: Train Classifier
def train_classifier(X_train_tfidf, y_train):
    # Initialize a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()

    # Train the classifier on the TF-IDF transformed training data and their corresponding labels
    classifier.fit(X_train_tfidf, y_train)

    return classifier

# Step 4: Evaluate Classifier
def evaluate_classifier(classifier, X_test_tfidf, y_test):
    # Predict labels for the test data
    y_pred = classifier.predict(X_test_tfidf)

    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    confusion_mat = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, precision, recall, f1, confusion_mat, class_report

if __name__ == "__main__":
    # Load data
    df = load_data('spam.csv')
    
    # Preprocess data
    X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess_data(df)
    
    # Train classifier
    classifier = train_classifier(X_train_tfidf, y_train)
    
    # Evaluate classifier
    accuracy, precision, recall, f1, confusion_mat, class_report = evaluate_classifier(classifier, X_test_tfidf, y_test)
    
    # Print out evaluation results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{confusion_mat}')
    print(class_report)
