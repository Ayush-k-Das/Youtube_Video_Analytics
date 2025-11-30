# 🎬 **YouTube Video Performance Analytics Dashboard & Engagement Rate Prediction (ML)**

This project combines **data analytics**, **Excel dashboarding**, and **machine learning** to analyze more than **537 YouTube videos** and predict their **engagement rate** using a Random Forest model.

It includes:

* A fully interactive **YouTube analytics dashboard** created in Excel
* KPI summaries (views, engagement, likes-to-view ratio & more)
* Top-10 performance charts
* A machine learning model to **predict engagement rate** based on video metadata

---

## 📊 **Dashboard Overview**

The Excel dashboard visualizes the dataset using KPIs and performance insights such as:

### **🔹 Key Metrics**

* **Average View Count**
* **Average Engagement Rate**
* **Average Likes-to-View Ratio**
* **Average Video Age (days)**
* **Average Duration (seconds)**

### **🔹 Visual Analytics**

* View Count by Category ID
* Top 10 YouTube Channels by View Count
* Top 10 Channels by Like Count
* Top 10 Channels by Comment Count
* Top 10 Average Engagement Rate by Channel
* More performance patterns across categories

## 🤖 **Engagement Rate Prediction (Machine Learning)**

This project uses a **Random Forest Regressor** to predict engagement rate (`likes + comments / views`) based on video features such as:

* Duration
* Category
* View Count
* Like Count
* Comment Count
* Video Age
* and other metadata

### **🔧 Model Code**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R²:", r2)
```

---

## 📈 **Model Performance**

| Metric                        | Score          |
| ----------------------------- | -------------- |
| **Mean Absolute Error (MAE)** | **0.00097975** |
| **R² Score**                  | **0.983078**   |

✔ High R² indicates strong predictive accuracy.
✔ Very low MAE shows minimal error in predicting engagement rate.

---

## 📂 **Project Structure**

```
/
├── dataset/             # Cleaned YouTube dataset (not included in repo)
├── dashboard/           # Excel dashboard file (.xlsx)
├── ml-model/            # ML notebooks or .py files
├── scripts/             # Data preprocessing scripts
└── README.md            # Project documentation
```

---

## ▶️ **How to Run the Model**

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the ML script**

```bash
python engagement_rate_model.py
```

### **3. Open the Excel dashboard**

The dashboard file is located under the `/dashboard` folder.

---

## 🚀 **Features**

* Full YouTube analytics dashboard
* Automated engagement rate prediction
* Top-10 channel comparison
* Data cleaning + preprocessing workflow
* Export-ready visuals

---

## 🛠️ **Technologies Used**

* **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib)
* **Excel** (PivotTables, Charts, Slicers)
* **Machine Learning** (RandomForestRegressor)
* **Data Visualization**
---

## ⭐ **Contributions**

Feel free to fork the repo and contribute improvements via pull requests!

---

## 📜 **License**

This project is released under the MIT License.

---

Just tell me!
