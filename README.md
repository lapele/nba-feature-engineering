# NBA Player Longevity Prediction — Feature Engineering

**Author:** Lapele Kenneth  
**Program:** 3MTT AI & Machine Learning Fellowship  
**Project Type:** Feature Engineering for Binary Classification

---

## Dataset

- **Source:** NBA Players Stats dataset
- **Size:** 1,340 players × 22 columns
- **File:** `nba_players.csv`
- **Task:** Predict whether an NBA player will remain in the league for **5 or more years** (`target_5yrs = 1`) based on early-career performance statistics

---

## Prediction Goal

Given a player's first-season statistics (games played, minutes, scoring, shooting percentages, rebounds, assists, steals, blocks, turnovers), predict long-term career longevity. This is a **binary classification problem:**
- `1` → player lasted 5+ years in the NBA
- `0` → player did not reach 5 years

**Class Balance:** ~62% of players in the dataset lasted 5+ years.

---

## Feature Engineering Workflow

### 1. Target Variable Isolation
`target_5yrs` is defined as the dependent variable (label). All remaining columns are potential predictors.

### 2. Noise Column Removal
| Column | Reason |
|--------|--------|
| `Unnamed: 0` | Row index — zero predictive value |
| `name` | Player identifier — causes data leakage |

### 3. Correlation Analysis & Multicollinearity Reduction
A Pearson correlation matrix was computed for all features. Pairs with |r| > 0.85 were flagged as redundant. Columns removed:

| Dropped | Retained | Correlation |
|---------|----------|-------------|
| `fgm`, `fga` | `fg` (FG%) | 0.98–0.99 with `pts` |
| `ftm`, `fta` | `ft` (FT%) | 0.98 with each other |
| `3p_made`, `3pa` | `3p` (3P%) | 0.98 with each other |
| `dreb`, `oreb` | `reb` | 0.93–0.98 with `reb` |

Shooting **percentage** metrics were preferred over raw count metrics as they normalise for playing time.

### 4. Engineered Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pts_per_min` | `pts / min` | Scoring efficiency per minute; removes playing-time bias |
| `efficiency_rating` | `pts + reb + ast + stl + blk - tov` | Composite all-round impact metric |
| `ast_tov_ratio` | `ast / (tov + 0.001)` | Basketball IQ and ball security indicator |

### 5. Missing Value Handling
No missing values existed in the original dataset. Any nulls introduced during feature engineering (e.g., division by zero in `pts_per_min`) were imputed using **column median** — robust to outliers and bias-free.

---

## Repository Contents

```
├── nba_feature_engineering.ipynb   # Main notebook — all cells executed
├── nba_players.csv                 # Raw input dataset
├── nba_clean_features.csv          # Final model-ready dataset (output)
├── correlation_heatmap.png         # Feature correlation heatmap
├── engineered_features_dist.png    # Distribution plots of new features
└── README.md                       # This file
```

---

## Final Dataset Summary

| Metric | Value |
|--------|-------|
| Rows | 1,340 |
| Features (X) | 14 |
| Target column | `target_5yrs` |
| Null values | 0 |
| Engineered features | 3 (`pts_per_min`, `efficiency_rating`, `ast_tov_ratio`) |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nba-feature-engineering.git
cd nba-feature-engineering

# Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook nba_feature_engineering.ipynb
```

---

## Tools Used
- **Python 3.10**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **seaborn / matplotlib** — visualisation
- **Jupyter Notebook** — interactive development environment
