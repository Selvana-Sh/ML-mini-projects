# ML-mini-projects
> Five mini machine learning projects covering classification, regression, and prediction tasks using real-world datasets.

---

## Table of Contents

1. [Fraud Detection](#1-fraud-detection)
2. [Tip Prediction](#2-tip-prediction)
3. [Sales Prediction](#3-sales-prediction)
4. [Product Demand Prediction](#4-product-demand-prediction)
5. [Stock Price Prediction](#5-stock-price-prediction)

---

## 1. Fraud Detection

### Introduction

This project builds a binary classifier to detect fraudulent financial transactions. Given a transaction's type, amount, and account balance information, the model predicts whether it is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`). Fraud detection is a critical real-world problem where even a small improvement in accuracy can prevent significant financial loss.

### Getting Started

**Dataset:** `fraud_log.csv` — 6,362,620 transactions with 11 columns including transaction type, amount, origin/destination balances, and a fraud label.

**Download Dataset:** [Online Payments Fraud Detection — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)

> After downloading, create a folder called `data/` in your project directory and place `fraud_log.csv` inside it. Then update the loading line in the notebook to:
> ```python
> df = pd.read_csv("data/fraud_log.csv")
> ```

**Required Libraries:**
```
numpy, pandas, plotly, scikit-learn
```

Install with:
```bash
pip install numpy pandas plotly scikit-learn
```

### How to Run

```bash
jupyter notebook fraud.ipynb
```

Run all cells top to bottom. The dataset path may need to be updated to your local path in the `pd.read_csv(...)` cell.

### Libraries & Functions Used

| Library / Function | Purpose |
|---|---|
| `pandas` | Loading and cleaning the dataset (`read_csv`, `get_dummies`, `drop`) |
| `numpy` | Generating random indices for sample testing |
| `sklearn.model_selection.train_test_split` | Splitting data into 80% train / 20% test |
| `sklearn.tree.DecisionTreeClassifier` | The classification model |
| `sklearn.metrics.accuracy_score` | Evaluating classification accuracy |
| `plotly.express.pie` | Visualizing the fraud vs. non-fraud class distribution |

### Step-by-Step Output

**Step 1 — Load Data**
```
   step      type    amount     nameOrig  oldbalanceOrg  newbalanceOrig  ...
0     1   PAYMENT   9839.64  C1231006815       170136.0       160296.36
1     1   PAYMENT   1864.28  C1666544295        21249.0        19384.72
...
RangeIndex: 6,362,620 entries | 11 columns | No missing values
```

**Step 2 — Preprocessing**
- Dropped non-numeric identifier columns: `nameOrig`, `nameDest`
- One-hot encoded the `type` column (PAYMENT, TRANSFER, CASH_OUT, etc.) using `get_dummies`
- Defined features `X` (all columns except `isFraud`) and target `y` (`isFraud`)

**Step 3 — Train/Test Split**
- 80% training / 20% test, `random_state=42`

**Step 4 — Train Model**
```
DecisionTreeClassifier(random_state=42)
```

**Step 5 — Sample Prediction**
```
Random Index: 322946
Prediction:   [0]   → Not Fraud
Actual Value:  0    → Not Fraud ✓
```

**Step 6 — Visualization**
- Pie chart showing the class imbalance between fraud and non-fraud transactions

### Results

| Metric | Value |
|---|---|
| **Model Accuracy (`model.score`)** | **99.97%** |
| Model Type | Decision Tree Classifier |
| Train/Test Split | 80% / 20% |
| Dataset Size | 6,362,620 rows |

The Decision Tree Classifier achieves near-perfect accuracy. The high score reflects both the model's strength and the dataset's clear distinguishing patterns between fraudulent and legitimate transactions.

---

## 2. Tip Prediction

### Introduction

This project predicts the tip amount a customer will leave at a restaurant based on features like the total bill, party size, day of the week, time of day, and whether the customer is a smoker. It uses a Linear Regression model to learn the relationship between these variables and the tip amount.

### Getting Started

**Dataset:** `tips.csv` — 244 restaurant transactions with columns: `total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size`.

**Download Dataset:** [Waiter Tips Dataset — GitHub Raw](https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv)

> After downloading, create a folder called `data/` in your project directory and place `tips.csv` inside it. Then update the loading line in the notebook to:
> ```python
> df = pd.read_csv("data/tips.csv")
> ```

**Required Libraries:**
```
numpy, pandas, matplotlib, seaborn, plotly, scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
```

### How to Run

```bash
jupyter notebook tip.ipynb
```

Run all cells top to bottom. No external path changes needed if `tips.csv` is in the same directory.

### Libraries & Functions Used

| Library / Function | Purpose |
|---|---|
| `pandas` | Loading data, encoding categoricals (`map`, `get_dummies`) |
| `numpy` | Numerical operations |
| `matplotlib.pyplot` | Plotting the correlation heatmap |
| `seaborn` | Styling the heatmap (`heatmap`) |
| `plotly.express` | Interactive scatter and pie charts |
| `sklearn.linear_model.LinearRegression` | The regression model |
| `sklearn.model_selection.train_test_split` | 80/20 data split |

### Step-by-Step Output

**Step 1 — Load Data**
```
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
...
244 rows × 7 columns | No missing values
```

**Step 2 — Preprocessing**
- Mapped `sex`: Male → 0, Female → 1
- Mapped `smoker`: No → 0, Yes → 1
- One-hot encoded `day` and `time` using `get_dummies(drop_first=True)`

**Step 3 — After Encoding**
```
   total_bill   tip  sex  smoker  size  day_Sat  day_Sun  day_Thur  time_Lunch
0       16.99  1.01    1       0     2    False     True     False       False
...
```

**Step 4 — Train/Test Split + Train**
- Features `X`: all columns except `tip` | Target `y`: `tip`
- 80% train / 20% test, `random_state=42`

**Step 5 — Visualization**
- Correlation heatmap (Oranges colormap) — `total_bill` has the strongest correlation with `tip`

### Results

| Metric | Value |
|---|---|
| **Model R² Score (`model.score`)** | **0.4373** |
| Model Type | Linear Regression |
| Train/Test Split | 80% / 20% |
| Dataset Size | 244 rows |

An R² of 0.44 means the model explains about 44% of the variance in tip amounts. This is expected — tipping behavior is influenced by subjective factors (mood, service quality) not captured in the dataset. `total_bill` is the strongest predictor.

---

## 3. Sales Prediction

### Introduction

This project predicts product sales based on advertising spend across three channels: TV, Radio, and Newspaper. Using a Linear Regression model, it quantifies how much each advertising medium contributes to sales, helping businesses allocate their marketing budget more effectively.

### Getting Started

**Dataset:** `advertising.csv` — 200 records with columns: `TV`, `Radio`, `Newspaper`, `Sales`.

**Download Dataset:** [Future Sales Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/ashydv/advertising-dataset?resource=download)

> After downloading, create a folder called `data/` in your project directory and place `advertising.csv` inside it. Then update the loading line in the notebook to:
> ```python
> df = pd.read_csv("data/advertising.csv")
> ```

**Required Libraries:**
```
numpy, pandas, matplotlib, seaborn, plotly, scikit-learn, statsmodels
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn statsmodels
```

### How to Run

```bash
jupyter notebook sales_prediction.ipynb
```

Run all cells top to bottom. Update the CSV path if needed.

### Libraries & Functions Used

| Library / Function | Purpose |
|---|---|
| `pandas` | Loading and exploring the dataset |
| `numpy` | Numerical support |
| `matplotlib.pyplot` | Actual vs. predicted scatter plot |
| `seaborn` | Statistical visualizations |
| `plotly.express.scatter` | Interactive scatter plots with OLS trendlines |
| `statsmodels` | Required by Plotly for OLS trendline rendering |
| `sklearn.linear_model.LinearRegression` | The regression model |
| `sklearn.model_selection.train_test_split` | 80/20 data split |

### Step-by-Step Output

**Step 1 — Load Data**
```
       TV  Radio  Newspaper  Sales
0   230.1   37.8       69.2   22.1
1    44.5   39.3       45.1   10.4
2    17.2   45.9       69.3   12.0
...
200 rows × 4 columns
```

**Step 2 — Define Features & Target**
- Features `X`: `TV`, `Radio`, `Newspaper`
- Target `y`: `Sales`

**Step 3 — Train/Test Split + Train**
- 80% train / 20% test, `random_state=42`
- Fitted `LinearRegression()` on training data

**Step 4 — Sample Predictions**
```
   Actual  Predicted
0    16.9  17.034772
1    22.4  20.409740
2    21.4  23.723989
3     7.3   9.272785
4    24.7  21.682719
```

**Step 5 — Visualizations**
- Interactive scatter: TV vs Sales, Radio vs Sales, Newspaper vs Sales (each with OLS trendline)
- Static scatter: Actual vs Predicted Sales

### Results

| Metric | Value |
|---|---|
| **Model R² Score (`model.score`)** | **0.9059** |
| Model Type | Linear Regression |
| Train/Test Split | 80% / 20% |
| Dataset Size | 200 rows |

An R² of 0.91 indicates the model explains 91% of the variance in sales — a strong result. TV advertising has the strongest linear relationship with sales, followed by Radio. Newspaper advertising shows minimal impact.

---

## 4. Product Demand Prediction

### Introduction

This project predicts the number of units sold for a retail product based on store, pricing, and discount information. It compares a Decision Tree Regressor against a Random Forest Regressor, finding that the ensemble method better captures the complex, noisy relationship between price and demand.

### Getting Started

**Dataset:** `demand.csv` — retail product records with columns: `ID`, `Store ID`, `Total Price`, `Base Price`, `Units Sold`.

**Download Dataset:** [Product Demand Prediction Dataset — GitHub Raw](https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv)

> After downloading, create a folder called `data/` in your project directory and place `demand.csv` inside it. Then update the loading line in the notebook to:
> ```python
> df = pd.read_csv("data/demand.csv")
> ```

**Required Libraries:**
```
numpy, pandas, matplotlib, seaborn, plotly, scikit-learn, statsmodels
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn statsmodels
```

### How to Run

```bash
jupyter notebook product_demand.ipynb
```

Run all cells top to bottom. Update the CSV path if needed.

### Libraries & Functions Used

| Library / Function | Purpose |
|---|---|
| `pandas` | Loading data, null handling (`dropna`), feature creation |
| `numpy` | Numerical support |
| `matplotlib.pyplot` | Actual vs. predicted scatter plot |
| `seaborn` | Correlation heatmap |
| `plotly.express.scatter` | Interactive price vs. demand plots with trendlines |
| `statsmodels` | Required by Plotly for OLS trendlines |
| `sklearn.tree.DecisionTreeRegressor` | Primary regression model |
| `sklearn.ensemble.RandomForestRegressor` | Comparison ensemble model |
| `sklearn.model_selection.train_test_split` | 80/20 data split |

### Step-by-Step Output

**Step 1 — Load Data**
```
   ID  Store ID  Total Price  Base Price  Units Sold
0   1      8091      99.0375    111.8625          20
1   2      8091      99.0375     99.0375          28
...
1 missing value in Total Price → removed with dropna()
```

**Step 2 — Correlation Matrix**
```
             Total Price  Base Price  Units Sold
Total Price     1.000000    0.958885   -0.235625
Base Price      0.958885    1.000000   -0.140032
Units Sold     -0.235625   -0.140032    1.000000
```

**Step 3 — Feature Engineering**
- Created `Discount = Base Price − Total Price`
- Created `Discount Percent = (Discount / Base Price) × 100`
- These engineered features help the model capture price-sensitivity effects on demand

**Step 4 — Features & Target**
- Features `X`: `Store ID`, `Total Price`, `Base Price`, `Discount Percent`
- Target `y`: `Units Sold`

**Step 5 — Sample Predictions (Decision Tree)**
```
   Actual   Predicted
0      41   35.554196
1      13   35.554196
2     339  182.204313
3      14   15.023276
4       4   35.554196
```

**Step 6 — Visualizations**
- Scatter: Actual vs. Predicted Units Sold
- Interactive: Total Price vs. Units Sold (OLS trendline)
- Interactive: Base Price vs. Units Sold (OLS trendline)

### Results

| Model | R² Score |
|---|---|
| Decision Tree Regressor | 0.3228 |
| **Random Forest Regressor** | **0.6116** |

| Setting | Value |
|---|---|
| Decision Tree `max_depth` | 5 |
| Decision Tree `min_samples_split` | 10 |
| Decision Tree `min_samples_leaf` | 5 |
| Train/Test Split | 80% / 20% |

The Random Forest significantly outperforms the single Decision Tree (R² 0.61 vs 0.32). Product demand is inherently noisy — customer behavior and store-level factors contribute to unpredictability that ensemble methods handle better.

---

## 5. Stock Price Prediction

### Introduction

This project predicts the next-day closing price of Microsoft (MSFT) stock using one year of historical data fetched live from Yahoo Finance. It uses a Random Forest Regressor with lag features — the previous two days' closing prices — to capture the autocorrelation inherent in financial time series data.

### Getting Started

**Data Source:** Yahoo Finance via `yfinance` — MSFT daily OHLCV data, 2025-03-05 to 2026-03-05 (~252 trading days).

> **No dataset download needed for this project.** Data is fetched directly inside the notebook using the `yfinance` library with a single line:
> ```python
> df = yf.download("MSFT", start="2025-03-05", end="2026-03-05")
> ```
> This is the recommended approach for stock data — it always pulls the most up-to-date prices and removes the need to manage CSV files manually. Just make sure you have an internet connection when running the notebook.

**Required Libraries:**
```
numpy, pandas, matplotlib, scikit-learn, yfinance
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn yfinance
```

### How to Run

```bash
jupyter notebook stock_prediction.ipynb
```

Run all cells top to bottom. Data is fetched automatically — internet connection required.

### Libraries & Functions Used

| Library / Function | Purpose |
|---|---|
| `yfinance.download` | Fetching live MSFT OHLCV data from Yahoo Finance |
| `pandas` | Data manipulation, lag feature creation with `.shift()` |
| `numpy` | Numerical support |
| `matplotlib.pyplot` | Line chart of actual vs. predicted prices over time |
| `sklearn.ensemble.RandomForestRegressor` | Ensemble regression model (chosen over Decision Tree for better time-series performance) |
| `sklearn.model_selection.train_test_split` | Chronological 80/20 split (no shuffle to prevent data leakage) |

### Step-by-Step Output

**Step 1 — Load Data from Yahoo Finance**
```
Price            Close        High         Low        Open    Volume
Ticker            MSFT        MSFT        MSFT        MSFT      MSFT
Date                                                                
2025-03-05  397.973267  398.618352  385.856040  386.382012  23433100
2025-03-06  393.874664  399.094681  389.696628  391.284400  ...
```

**Step 2 — Lag Feature Engineering**
```python
df["Close_Lag1"] = df["Close"].shift(1)  # Yesterday's closing price
df["Close_Lag2"] = df["Close"].shift(2)  # Two days ago closing price
df = df.dropna().reset_index()           # Remove NaN rows from shifting
```

**Step 3 — Features & Target**
- Features `X`: `Open`, `High`, `Low`, `Volume`, `Close_Lag1`, `Close_Lag2`
- Target `y`: `Close` (current day's closing price)

**Step 4 — Time-Series Split (No Shuffle)**
```python
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]   # First 80% chronologically
X_test  = X.iloc[split_index:]   # Last 20% chronologically
```
Shuffling is deliberately avoided to prevent future data leaking into training.

**Step 5 — Train Model**
```
RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
```

**Step 6 — Visualization**
- Line chart: Actual closing price (blue) vs. Predicted closing price (red) across the test period

### Results

| Metric | Value |
|---|---|
| **Model R² Score (`model.score`)** | **0.9685** |
| Model Type | Random Forest Regressor |
| Estimators | 200 trees |
| Max Depth | 12 |
| Train/Test Split | 80% / 20% (chronological) |
| Data Source | Yahoo Finance via `yfinance` |

An R² of 0.9685 means the model explains ~97% of the variance in next-day closing prices. The Random Forest was chosen over a single Decision Tree because ensemble averaging reduces overfitting, which is critical for the noisy dynamics of stock market data. The lag features (`Close_Lag1`, `Close_Lag2`) are the strongest predictors, capturing day-to-day price autocorrelation.

---

*All projects follow the same general pipeline: Load Data → Explore & Clean → Engineer Features → Train Model → Evaluate → Visualize.*
