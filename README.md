# Movie Rating Prediction 🎬 &nbsp; [![View Code](https://img.shields.io/badge/Jupyter-View_Notebook-orange?logo=jupyter)](Movie%20Rating%20Prediction%20(1).ipynb)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success)

> **Predicting IMDb movie ratings for Indian cinema using machine learning — achieving an RMSE of 0.867 with Random Forest regression on 15,500+ movies.**

<br>

<p align="center">
  <img src="https://img.shields.io/badge/🎬_Movies_Analyzed-15,508-blue?style=for-the-badge" alt="Movies"/>
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/📊_RMSE-0.867-green?style=for-the-badge" alt="RMSE"/>
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/🔧_Features_Engineered-7-orange?style=for-the-badge" alt="Features"/>
</p>

<br>

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Author](#author)

<br>

## Problem Statement

Can we predict how audiences will rate a movie before it's released? This project tackles that question by building a **regression model** that estimates IMDb ratings for Indian movies based on metadata such as genre, director, cast, duration, and vote count. Accurate rating predictions can help studios, distributors, and streaming platforms make data-driven decisions about content acquisition and marketing.

<br>

## Dataset

| Property | Detail |
|----------|--------|
| **Source** | [IMDb Movies India](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) |
| **Size** | 15,509 movies |
| **Features** | 10 columns — Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1–3 |
| **Target** | `Rating` (continuous, 1–10 scale) |
| **Challenges** | ~49% missing ratings, missing durations, non-numeric vote counts |

<br>

## Methodology

### 1. Data Preprocessing
- Extracted numeric year from parenthesized strings (e.g., `(2019)` → `2019`)
- Parsed duration by stripping `min` suffix and converting to integer
- Filled missing values: **median** for Year, Duration, Votes; **mode** for Genre; `"Unknown"` for Director/Actors
- Converted `Votes` from string to numeric using `pd.to_numeric(errors='coerce')`
- **Imputed missing ratings** using a Linear Regression model trained on Year, Duration, and Votes

```python
# Impute missing ratings with a predictive model
model = LinearRegression()
train_data = movie_data.dropna(subset=['Rating'])
model.fit(train_data[['Year', 'Duration', 'Votes']], train_data['Rating'])

missing_data = movie_data[movie_data['Rating'].isnull()]
predicted_ratings = model.predict(missing_data[['Year', 'Duration', 'Votes']])
movie_data.loc[movie_data['Rating'].isnull(), 'Rating'] = predicted_ratings
```

### 2. Feature Engineering
- **Genre Count** — number of genres per movie (`len(genre.split(', '))`)
- **Movie Age** — `2024 - Year` to capture recency effects
- **One-Hot Encoding** — applied to Genre via `ColumnTransformer` with `handle_unknown='ignore'`

### 3. Model Training
- **Algorithm:** Random Forest Regressor (100 estimators)
- **Pipeline:** `ColumnTransformer` (OneHotEncoder + passthrough numerics) → `RandomForestRegressor`
- **Split:** 80/20 train-test split (`random_state=42`)

```python
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_pipeline.fit(X_train, y_train)
```

<br>

## Key Results

| Metric | Value |
|--------|-------|
| **Root Mean Squared Error (RMSE)** | **0.867** |

The model predicts movie ratings within roughly **±0.87 points** on the 1–10 IMDb scale, demonstrating that metadata features like genre, duration, votes, and movie age carry meaningful signal for rating prediction.

### Feature Inputs

| Feature Type | Features Used |
|-------------|---------------|
| **Numeric** | Year, Duration, Votes, Genre_Count, Movie_Age |
| **Categorical** | Genre (one-hot encoded) |

<br>

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation & cleaning |
| NumPy | Numerical operations |
| Scikit-learn | ML pipeline, Random Forest, metrics |
| Jupyter Notebook | Interactive development |

<br>

## How to Run

```bash
# Clone the repository
git clone https://github.com/ouyale/Movie-Rating-Prediction-v2.git
cd Movie-Rating-Prediction-v2

# Install dependencies
pip install pandas numpy scikit-learn jupyter

# Launch the notebook
jupyter notebook "Movie Rating Prediction (1).ipynb"
```

<br>

## Author

**Barbara Obayi** — Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-ouyale-181717?logo=github)](https://github.com/ouyale)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Barbara_Obayi-0A66C2?logo=linkedin)](https://www.linkedin.com/in/barbara-weroba-obayi31/)
[![Portfolio](https://img.shields.io/badge/Portfolio-ouyale.github.io-4fc3f7)](https://ouyale.github.io)

---
