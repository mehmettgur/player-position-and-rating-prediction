# Player Position & Rating Prediction

This project uses the **European Soccer Database (SQLite)** to:

1. Extract players' **average on-field coordinates** from matches.
2. Generate position labels using **K-Means** and **rule-based** methods (4 general roles, 7 detailed roles).
3. Predict players' **overall rating** using position-enriched features with Ridge, ElasticNet, RandomForest, and XGBoost regressors.
4. Create visualizations for model outputs and **feature importances**.

Project pipeline: position detection → position-aware rating prediction → evaluation and visualization

---

## 📂 Project Structure

```
src/
  soccer_rating/
    data/          # SQLite reading, parquet I/O
    features/      # Average coordinate calculation
    labeling/      # KMeans and rule-based position labeling
    modeling/      # M1–M8 frameworks, model training
    evaluation/    # Metric calculation
    viz/           # plots.py — visualization of results and feature importance
scripts/
  build_positions.py
  label_positions.py
  merge_player_data.py
  add_rule_labels.py
  train_models.py
  evaluate_models.py
  feature_importance.py
requirements.txt
pyproject.toml
```

---

## 📊 Model Configurations (M1–M8)

| Model | Label | Role Level | Feature Set |
|---|---|---|---|
| M1 | kmeans_position_label | General (4 roles) | Selected |
| M2 | kmeans_position_label | General | All |
| M3 | kmeans_detailed_position | Detailed (7 roles) | Selected |
| M4 | kmeans_detailed_position | Detailed | All |
| M5 | rule_position_label | General | Selected |
| M6 | rule_position_label | General | All |
| M7 | rule_detailed_position | Detailed | Selected |
| M8 | rule_detailed_position | Detailed | All |

---

## 🔍 Step-by-Step Workflow

### 1. Extract Average Coordinates
```bash
python scripts/build_positions.py --sqlite <path_to_database.sqlite> --out data/positions.parquet
```
- Computes per-player average `avg_X` and `avg_Y` positions from `home/away_player_Xi/Yi` columns in the `Match` table.

### 2. Position Labeling with K-Means
```bash
python scripts/label_positions.py --positions data/positions.parquet --out data/labels.parquet
```
- Runs KMeans (n_clusters=4 and n_clusters=7) to produce **general** (`kmeans_position_label`) and **detailed** (`kmeans_detailed_position`) labels.

### 3. Merge with Player Attributes
```bash
python scripts/merge_player_data.py --sqlite <path_to_database.sqlite> --positions data/labels.parquet --out data/player_data.parquet
```
- Merges with the most recent `Player_Attributes` records to obtain features.

### 4. Rule-Based Labeling
```bash
python scripts/add_rule_labels.py --data_in data/player_data.parquet --data_out data/player_data_rule.parquet
```
- Uses rule-based functions to create **general** (`rule_position_label`) and **detailed** (`rule_detailed_position`) labels.

### 5. Model Training
```bash
python scripts/train_models.py --data data/player_data_rule.parquet --out runs/local_test
```
- Trains position-specific regressors for all M1–M8 configurations and 4 different regressors.
- Output: `runs/local_test/results.csv`

### 6. Model Evaluation
```bash
python scripts/evaluate_models.py --runs_dir runs
```
- Outputs weighted average RMSE and R² scores for each model.

### 7. Feature Importance Calculation
```bash
python scripts/feature_importance.py --data data/player_data_rule.parquet --out figs --model M6
```
- For the selected model (default Ridge), computes per-position coefficients or feature importances.
- CSV output can be used for visualizations.

### 8. Visualization (`plots.py`)
- `src/soccer_rating/viz/plots.py` contains functions to plot training results and feature importance.
- Example: `plot_model_comparison()` → RMSE/R² comparison of models
- Example: `plot_feature_importance()` → top 5 features per position

![model-comparison](figs/model_comparison.png)
*Model comparison*

![feature-importance](figs/feature_importance.png)
*Top features by position*

---

## 🛠 Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (for src layout)
pip install -e .
```
---

## 🚀 Example Run (Windows PowerShell)
```powershell
$Env:ESD_SQLITE="C:\path	o\database.sqlite"

python scriptsuild_positions.py   --sqlite $Env:ESD_SQLITE --out data\positions.parquet
python scripts\label_positions.py   --positions data\positions.parquet --out data\labels.parquet
python scripts\merge_player_data.py --sqlite $Env:ESD_SQLITE --positions data\labels.parquet --out data\player_data.parquet
python scriptsdd_rule_labels.py   --data_in data\player_data.parquet --data_out data\player_data_rule.parquet
python scripts	rain_models.py      --data data\player_data_rule.parquet --out runs\local_test
python scripts\evaluate_models.py   --runs_dir runs
python scriptseature_importance.py --data data\player_data_rule.parquet --out figs --model M6
```

---

## 📄 Related Documents

- **project_report.pdf** → The main project paper explaining the methodology, experiments, and results in detail.
- **report_appendix.pdf** → Additional appendix with supporting figures, tables, and extended data used in the project.

These files are included in the repository for reference and further reading.

> The `figs/` folder contains visual outputs matching the examples shown in the README.
