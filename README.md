# 📘 DeveloperHub Task 1 – Iris Dataset EDA and Visualization

## 📌 Task Objective
This internship task focuses on exploring and visualizing the **Iris dataset** to develop a strong understanding of data analysis, visualization, and basic machine learning model evaluation using Python.

---

## 📁 Dataset
- **Name**: Iris Dataset  
- **Source**: Built-in dataset via `seaborn.load_dataset('iris')`  
- **Features**:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
  - `species` (target class: Setosa, Versicolor, Virginica)

---

## 🛠️ Tools & Libraries Used
- **Pandas** – for data handling and manipulation  
- **Seaborn** – for advanced visualizations  
- **Matplotlib** – for plotting  
- **Scikit-learn** – for ML modeling and evaluation  
- **Streamlit** – for deploying the prediction app  

---

## 🚀 Approach

### 🔍 1. Dataset Loading & Inspection
- Used `pandas` and `seaborn` to load the dataset  
- Checked `.shape`, `.info()`, and `.head()` to understand the structure  

### 🧹 2. Data Cleaning & Preprocessing
- Verified data types  
- Confirmed no missing/null values  

### 📊 3. Exploratory Data Analysis (EDA)
- **Pairplot** to show feature relationships  
- **Histograms** for distribution analysis  
- **Box plots** to detect outliers and spread  

### 🤖 4. Model Training & Testing
- Used **Random Forest Classifier** from `sklearn`  
- Performed train-test split (80/20)  

### 🧪 5. Model Evaluation
- Calculated **accuracy score**  
- Displayed **confusion matrix**  
- Viewed **classification report**

### 🌐 6. Deployment
- Created a **Streamlit app** with:
  - Custom styled background 🌸  
  - Sliders for input  
  - Species image and description  
- Live on Streamlit Cloud

---

## 📊 Results & Insights
- **Petal length and width** are highly indicative of species  
- Random Forest achieved **~100% accuracy**  
- Dataset was **clean and well-structured**  

---

## ✅ Conclusion
This task demonstrated the effectiveness of **EDA**, the importance of **data visualization**, and how to deploy a simple ML model using **Streamlit**. It helped reinforce both **analytical** and **technical** skills in data science.

---

## 🎥 YouTube Video
📺 Watch the full walkthrough:  
[https://youtu.be/NOLx5sOpMso](https://youtu.be/NOLx5sOpMso)

---

## 🌐 Live App
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iris-data-exploration-eda-mrjbpudohvldczjbjilens.streamlit.app/)

---

## 📚 Useful Links
- [Seaborn Documentation](https://seaborn.pydata.org/)  
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [Matplotlib Documentation](https://matplotlib.org/)  

---

> 🔖 Submitted as part of the **Developer Hub Internship Program**
