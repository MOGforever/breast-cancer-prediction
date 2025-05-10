import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_wine, load_breast_cancer

def load_and_prepare_data(dataset_loader, test_size=0.3, random_state=42):
    """Load and prepare dataset for classification."""
    try:
        data = dataset_loader()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names, X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train a model and evaluate its performance."""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        return accuracy
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return None

def plot_results(x_data, y_data, x_label, y_label, title, filename, ylim=(0, 1)):
    """Plot and save results to a file."""
    plt.figure(figsize=(10, 6))
    plt.bar(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

# Part 1: Wine Classification
print("===== WINE CLASSIFICATION =====")
result = load_and_prepare_data(load_wine)
if result:
    X_train, X_test, y_train, y_test, _, _, _ = result
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    for name, model in models.items():
        train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)

# Subtask 1: Effect of Changing Random State
print("\n===== SUBTASK 1: EFFECT OF CHANGING RANDOM STATE =====")
random_states = [42, 0, 10, 20, 30, 100]
accuracies = []
for rs in random_states:
    dt = DecisionTreeClassifier(random_state=rs)
    accuracy = train_and_evaluate_model(dt, X_train, X_test, y_train, y_test, f"Decision Tree (random_state={rs})")
    accuracies.append(accuracy)

plot_results([str(rs) for rs in random_states], accuracies, 'Random State', 'Accuracy',
             'Effect of Random State on Decision Tree Accuracy', 'random_state_effect.png')

# Subtask 2: Breast Cancer Dataset Classification
print("\n===== SUBTASK 2: BREAST CANCER DATASET CLASSIFICATION =====")
result = load_and_prepare_data(load_breast_cancer)
if result:
    X_train_c, X_test_c, y_train_c, y_test_c, _, _, _ = result
    for name, model in models.items():
        train_and_evaluate_model(model, X_train_c, X_test_c, y_train_c, y_test_c, name)

# Subtask 3: Random Forest on Breast Cancer Dataset
print("\n===== SUBTASK 3: RANDOM FOREST ON BREAST CANCER DATASET =====")
result = load_and_prepare_data(load_breast_cancer)
if result:
    X_train_c, X_test_c, y_train_c, y_test_c, feature_names, X_cancer, y_cancer = result
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Tuned hyperparameters
    train_and_evaluate_model(rf, X_train_c, X_test_c, y_train_c, y_test_c, "Random Forest")

    # Cross-validation
    cv_scores = cross_val_score(rf, X_cancer, y_cancer, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Important Features - Random Forest')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    plt.close()