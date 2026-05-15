# NYC Airbnb Price Predictor

An end-to-end machine learning project predicting nightly Airbnb listing 
prices in New York City. Built to demonstrate a complete ML pipeline -- 
from raw data to model evaluation -- following the project structure from 
Hands-On Machine Learning (Aurélien Géron, Chapter 2).


## Overview

Given a listing's location, room type, availability, and host activity, 
can we predict its nightly price?

This project works through every step of a real-world ML workflow:
data exploration, preprocessing pipelines, model selection, 
hyperparameter tuning, and feature importance analysis -- using 
48,000+ real NYC Airbnb listings from 2019.


## Key Findings

- **Room type is the strongest price driver** -- whether a listing is 
  an entire home vs. a private room accounts for 31% of predictive power
- **Geography matters more than borough labels** -- raw latitude and 
  longitude outperformed neighbourhood_group as predictors, confirming 
  that price is continuous across space, not categorical by borough
- **Residual error is explainable** -- remaining prediction error stems 
  from unstructured signals (photos, host reputation, amenities) not 
  captured in structured data alone


## Results

| Model | CV RMSE |
|---|---|
| Linear Regression | 0.4361 |
| Decision Tree | 0.5660 |
| Random Forest | 0.4065 |
| Random Forest (tuned) | **0.4001** |

*RMSE on log-transformed price. Final test RMSE: 0.4001*


## Project Structure

```
nyc-airbnb-price-predictor/
│
├── nyc_airbnb_price_predictor.ipynb
├── data/                             
├── models/                           
└── README.md
```


## How to Run

**1. Clone the repo:**
```bash
git clone https://github.com/NinaweRahul/nyc-airbnb-price-predictor.git
```

**2. Download the dataset:**
Get `AB_NYC_2019.csv` from [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) 
and place it in the `data/` folder.

**3. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**4. Run the notebook:**
```bash
jupyter notebook nyc_airbnb_price_predictor.ipynb
```

Run all cells top to bottom. The trained model will be saved 
to `models/airbnb_price_predictor.pkl`.


## Tech Stack

- **Python** -- pandas, numpy, matplotlib, seaborn
- **Scikit-learn** -- Pipeline, ColumnTransformer, RandomForestRegressor, GridSearchCV
- **Jupyter Notebook**


## ML Pipeline Steps

1. Exploratory Data Analysis -- price distribution, geographic map, 
   correlation matrix
2. Data preparation -- outlier removal, log transformation, 
   sklearn Pipeline with ColumnTransformer
3. Model selection -- Linear Regression, Decision Tree, Random Forest
4. Evaluation -- cross-validation across all models
5. Hyperparameter tuning -- GridSearchCV on Random Forest
6. Feature importance analysis -- top 15 predictors visualized

---
*Dataset: [NYC Airbnb Open Data 2019](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) 
via Kaggle*
