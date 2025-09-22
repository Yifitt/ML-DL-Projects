# RandomForestFromScratch

**Description:**  
A simple Random Forest classifier implemented from scratch in Python **without using scikit-learn or any other ML library**.  
It uses multiple Decision Trees (implemented from scratch) and supports **random feature selection** and **bootstrap sampling** for training.

**Features:**  
- Ensemble of Decision Trees for improved accuracy.  
- Supports `max_depth`, `min_samples_split` and `n_features` per tree.  
- Majority voting for classification.  

**Usage Example:**

```python
from random_forest import RandomForest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest
clf = RandomForest(n_trees=20, max_depth=7)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
