# 🧪 Date-A-Scientist: Zodiac Sign Prediction on OKCupid Data

## Project Status

![Status Badge](https://img.shields.io/badge/Status-Complete-green)
![License](https://img.shields.io/badge/License-Codecademy%20Portfolio-blue)

## 📖 Overview

This portfolio project involves a comprehensive analysis of user data from OKCupid, a dating platform known for its detailed user profiles. The goal was to explore the wealth of information provided by users, including demographic details, lifestyle habits, and extensive free-text essays, by applying various Machine Learning techniques.

The primary objective was to practice formulating questions, implementing a full data science pipeline (EDA, Feature Engineering, Modeling), and critically evaluating the results, regardless of the model's performance.

---

## 🎯 Project Objectives & Research Questions

The central hypothesis was to determine if a user's **Zodiac Sign** could be predicted using non-astrological features extracted from their profiles.

### Research Questions:

1.  Can we accurately classify zodiac signs based on user profile data (demographics, lifestyle, and essay text)?
2.  Do Natural Language Processing (NLP) features, such as TF-IDF on essays, improve the predictive performance for classification tasks?
3.  What is the correlation strength between various profile features (e.g., age, height, drinking habits) and the target variable?
4.  What insights can be gained from applying Regression models to predict continuous variables (like `age`) using the same feature set?

---

## 💡 Key Findings and Critical Evaluation

The project’s most significant finding was the confirmation that **Zodiac Signs cannot be reliably predicted** from the available profile data.

| Task | Models Used | Best Performance | Conclusion |
| :--- | :--- | :--- | :--- |
| **Classification** (Zodiac) | LogReg, KNN, Random Forest, SVC | ~8.5% Accuracy | **Equivalent to Random Guessing** (8.33% baseline). |
| **Regression** (Age Prediction) | Linear Regression, Random Forest Regressor | $R^2 \approx 0.27$ | Shows weak predictive signal; features are insufficient for robust numerical prediction. |

**Critical Insight:**

The low performance across all models demonstrates a **low Signal-to-Noise Ratio** in the dataset regarding the astrological sign. The project concludes that the available features (lifestyle, demographics, and even basic essay text analysis) simply **do not encode the information necessary** to infer a user's zodiac sign, reinforcing the importance of **problem framing** in Data Science.

---

## 🛠️ Technical Stack & Methodology

The entire project pipeline was executed in a Jupyter Notebook, following standard machine learning practices:

* **Language:** Python 3
* **Data Manipulation:** `pandas`, `NumPy`
* **Visualization (EDA):** `matplotlib`, `seaborn` (Used for creating Histograms, Heatmaps, and Categorical Charts to justify feature selection).
* **Feature Engineering:** `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`) for Natural Language Processing on the essay features.
* **Modeling:** `scikit-learn` (Classification: `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`; Regression: `LinearRegression`, `RandomForestRegressor`).

---

## 🚀 Setup and Installation

To run this project locally, follow these steps:

### Prerequisites

* Python 3.x
* Jupyter Notebook
* Git (for version control)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gabrielarcangelbol/OKCupid-Date-A-Scientist-Starter]
    cd date-a-scientist
    ```

2.  **Install the necessary dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

3.  **Download the dataset:**
    The original data file (`okcupid_profiles.csv`) is provided in the Codecademy starter pack, which can be downloaded [here](https://content.codecademy.com/PRO/paths/data-science/OKCupid-Date-A-Scientist-Starter.zip).

4.  **Run the notebook:**
    ```bash
    jupyter notebook date-a-scientist.ipynb
    ```

---

## 📈 Future Enhancements

Based on the project's limitations, future work would focus on reframing the problem:

* **Reframed Task:** Pivot to predict more "learnable" and behavior-driven traits (e.g., predicting `smoker` or `diet` from essays).
* **Advanced NLP:** Utilize Deep Learning text embeddings (e.g., **BERT** or **Word2Vec**) to capture richer semantic context from the essays, moving beyond simple TF-IDF.
* **Unsupervised Learning:** Apply clustering techniques (K-Means) to identify underlying **Dating Archetypes** or user personas based on their lifestyle features.