# Customer Churn Analysis Using Machine Learning

A machine learning project that analyzes customer behavior and predicts customer churn using Python and Scikit-learn. The project includes data preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and customer segmentation using clustering techniques.

## Project Overview

Customer churn is an important business problem where companies lose customers over time. This project analyzes customer information to identify the factors that contribute to churn and builds a machine learning model to predict whether a customer is likely to leave.

The project also uses **K-Means Clustering** to segment customers based on their characteristics and behavior.

##  Objectives

* Analyze customer data and identify important churn-related factors.
* Perform data cleaning and preprocessing.
* Explore customer behavior using Exploratory Data Analysis (EDA).
* Prepare features for machine learning.
* Build and evaluate classification models.
* Predict whether a customer is likely to churn.
* Segment customers using K-Means clustering.
* Identify customer groups that may require targeted retention strategies.
* Deploy the project as an interactive Streamlit application.

##  Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning
* **Streamlit** – Web application and deployment
* **Joblib** – Saving and loading ML models
* **Jupyter Notebook / PyCharm** – Development environment

##  Machine Learning Techniques

### Classification

The project uses machine learning classification techniques to predict customer churn.

Typical workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Encoding & Scaling
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Churn Prediction
```

### Customer Segmentation

K-Means clustering is used to group customers with similar characteristics.

The clustering workflow includes:

```text
Customer Data
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
Elbow Method
     ↓
Optimal Number of Clusters
     ↓
K-Means Clustering
     ↓
Customer Segments
```

##  Exploratory Data Analysis

The project analyzes different customer attributes and their relationship with churn.

Key analysis areas include:

* Churn distribution
* Customer demographics
* Contract type
* Monthly charges
* Total charges
* Tenure
* Payment methods
* Services used
* Customer churn patterns

##  Model Evaluation

The machine learning model is evaluated using appropriate classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help determine how effectively the model identifies customers who are likely to churn.

##  Customer Segmentation

K-Means clustering is used to identify different customer segments.

The **Elbow Method** helps determine a suitable number of clusters by analyzing the Within-Cluster Sum of Squares (WCSS).

Customer segments can then be analyzed to understand:

* High-value customers
* Low-engagement customers
* High-risk customers
* Customers requiring retention strategies

##  Streamlit Dashboard

The project includes an interactive **Streamlit dashboard** where users can explore the analysis and make predictions.

### Dashboard Features

*  Customer churn overview
*  Interactive visualizations
*  Churn prediction
*  Customer segmentation
*  Model performance
*  Customer-level analysis
*  Business insights

##  Project Structure

```text
Customer-Churn-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── clustering.py
│
└── images/
    ├── dashboard.png
    ├── churn_analysis.png
    └── clustering.png
```

> Update the folder structure according to the actual files in your repository.

##  How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Customer-Churn-Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

The main Python libraries used in this project are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

A complete list of dependencies is available in:

```text
requirements.txt
```

##  Live Demo

🚀 **Streamlit App:**
Add your deployed Streamlit URL here.

```text
https://your-streamlit-app-url.streamlit.app
```

##  Project Screenshots

### Customer Churn Dashboard

Add your Streamlit dashboard screenshot here.

```text
![Customer Churn Dashboard](images/dashboard.png)
```

### Churn Analysis

```text
![Churn Analysis](images/churn_analysis.png)
```

### Customer Clustering

```text
![Customer Clustering](images/clustering.png)
```

##  Business Insights

The analysis can help businesses:

* Identify customers who have a high probability of churning.
* Understand the major factors associated with customer churn.
* Segment customers based on their characteristics.
* Develop targeted customer retention strategies.
* Improve customer satisfaction and loyalty.
* Reduce potential revenue loss caused by customer churn.

##  Future Improvements

* Add advanced ensemble models such as Random Forest and XGBoost.
* Implement hyperparameter tuning.
* Add real-time customer prediction.
* Improve the Streamlit dashboard with advanced filters.
* Add automated model retraining.
* Connect the application to a live database.
* Add explainable AI using SHAP.

##  Author

**Dhruv Zadafiya**

B.E. Information & Technology
Data Analytics | Machine Learning | Python

##  If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

##  License

This project is created for educational, internship, portfolio, and learning purposes.
