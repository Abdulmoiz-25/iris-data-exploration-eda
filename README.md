# 📘 DeveloperHub Task 1 – Iris Dataset EDA and Visualization

## 📌 Task Objective
This internship task focuses on exploring and visualizing the **Iris dataset** to develop a strong understanding of data analysis, visualization, and basic machine learning model evaluation using Python.

---

## 📁 Dataset
- **Name**: Iris Dataset
- **Source**: Built-in dataset available via `seaborn.load_dataset('iris')`
- **Features**:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
  - `species` (target class: Setosa, Versicolor, Virginica)

---

## 🛠️ Tools & Libraries Used
- `Pandas` – for data handling and manipulation  
- `Seaborn` – for advanced visualizations  
- `Matplotlib` – for plotting  
- `Scikit-learn` – for basic machine learning model and evaluation  

---

## 🚀 Approach
1. **Dataset Loading & Inspection**  
   Used `pandas` to load and inspect the structure using `.shape`, `.columns`, `.info()`, and `.head()`.

2. **Data Cleaning & Preprocessing**  
   Verified missing values and datatypes. The dataset was already clean.

3. **Exploratory Data Analysis (EDA)**  
   - Scatter plots for feature relationships (using `pairplot`)
   - Histograms for distribution
   - Box plots to detect outliers and understand feature spread

4. **Model Training & Testing**  
   - Applied a simple **Random Forest Classifier**
   - Used train-test split with 80/20 ratio

5. **Model Evaluation**  
   - Accuracy score
   - Confusion Matrix
   - Classification Report

---

## 📊 Results & Insights
- The species are clearly separable based on petal length and width.
- Random Forest achieved nearly **100% accuracy** on the test set.
- No missing or noisy data found in the dataset.

---

## ✅ Conclusion
This task demonstrated the power of **exploratory data analysis** using Python, highlighted the significance of **data visualization**, and introduced **basic machine learning evaluation**.

---

## 🎥 YouTube Video
Watch the full task walkthrough:  
🔗 [https://youtu.be/NOLx5sOpMso](https://youtu.be/NOLx5sOpMso)

---

## 🔗 Useful Links
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Docs](https://matplotlib.org/)

---

> 🔖 Submitted as part of the **Developer Hub Internship Program**
