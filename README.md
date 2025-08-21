# Gray: Machine Learning Model Tracker with MongoDB

Gray is a lightweight Python library that helps you **store, track, and retrieve machine learning models, hyperparameters, and performance metrics** using MongoDB.
It is designed to speed up experimentation by allowing you to automatically log model details and fetch the best hyperparameters for reuse.

---

## ✨ Features

* Store model details (name, hyperparameters, metrics, and tags) in MongoDB.
* Automatically extract and log hyperparameters & metrics from scikit-learn models.
* Retrieve the **best hyperparameters** based on chosen metrics (accuracy, F1-score, etc.).
* Tag-based filtering to match models with specific experimental contexts.
* Simple and extensible API.

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/gray.git
cd gray
pip install -r requirements.txt
```

**Requirements:**

* Python 3.8+
* pymongo
* scikit-learn
* numpy

Install dependencies manually if needed:

```bash
pip install pymongo scikit-learn numpy
```

---

## 🚀 Usage

### 1. Initialize the Gray Client

```python
from gray import Gray

gray = Gray(
    connection_string="mongodb://localhost:27017/",
    database="ml_tracking",
    collection="models"
)
```

---

### 2. Store Model Details Manually

```python
gray.store_model_details(
    model_name="RandomForestClassifier",
    hyperparameters={"n_estimators": 100, "max_depth": 10},
    performance_metrics={"accuracy": 0.92, "f1_score": 0.90},
    model_tags=["baseline", "random_forest"]
)
```

---

### 3. Automatically Store Model Details

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

gray.auto_store_model_details(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_tags=["iris", "random_forest"]
)
```

---

### 4. Retrieve Best Hyperparameters

```python
best_config = gray.get_best_hyperparameters(
    model_name="RandomForestClassifier",
    metric="accuracy",
    maximize=True,
    model_tags=["iris"]
)

print(best_config)
```

---

## 📖 Example

A sample **notebook (`test.ipynb`)** is included that demonstrates:

* Training a model.
* Logging it with Gray.
* Fetching the best hyperparameters.

---

## 🗂 Project Structure

```
.
├── gray.py          # Core Gray library
├── test.ipynb       # Example usage notebook
├── README.md        # Project documentation
```

---

## 🔮 Roadmap

* [ ] Add support for deep learning frameworks (PyTorch, TensorFlow).
* [ ] Add model versioning & experiment comparison.
* [ ] Add visualization dashboard for model performance.

---
