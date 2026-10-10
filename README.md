# Colombo District Land Price Prediction — AML Activity

**Goal:** Compare Random Forest and XGBoost regression models to estimate advertised land price per perch in Colombo District, Sri Lanka.

> Academic activity project, not a professional property valuation tool. Predictions are estimates based on this dataset and should not be used as verified market valuations.

## Dataset
- File: `data/colombo_land_prices.csv`
- Original source supplied for this project: [Kaggle — Land Prices of Colombo District, Sri Lanka](https://www.kaggle.com/datasets/ruchiraayeshmantha/land-prices-of-colombo-district)
- The submitted spreadsheet had 4,494 rows and 35 columns before cleaning.
- The dataset appears to contain advertisement records, not verified completed property transactions. Explain this limitation in the report.
- Review the original Kaggle page and dataset license/terms before redistribution.

## Target and leakage precautions
Target: `Price per Perch` (regression).

The model deliberately excludes:
- `Mentioned Price(Rs)`: directly related to the target and would cause target leakage.
- `Price_Scale`: a pricing representation/scale field related to how the price was quoted.
- `Address_ID`: identifier, not a meaningful property characteristic.
- `Posted_Date_new`: the supplied data spans a short posting period; the notebook extracts year/month features for exploration but does not use the raw date.
- `Price per Perch`: target itself.

`Address` is retained as a categorical location feature, with rare categories grouped by the preprocessing pipeline. The notebook removes exact duplicate rows and reports the count removed.

## Run in Google Colab
1. Upload this project folder to your GitHub repository, or upload `app.ipynb` and the CSV to Colab.
2. Open `app.ipynb` in Google Colab.
3. Run all cells from top to bottom.
4. If the dataset is not found automatically, the notebook asks you to upload `colombo_land_prices.csv`.
5. The notebook cleans the data, trains and evaluates both models, saves the winning model pipeline as `model.pkl`, and starts a Gradio interface.
6. Open the temporary Gradio URL printed by the final cell to test a single prediction or upload a CSV.

The Gradio URL only works while that Colab runtime is running. The notebook and source files are the reproducible submission; the temporary URL is not permanent hosting.

## Run locally (optional)
```bash
pip install -r requirements.txt
python app.py
```
`app.py` expects a compatible `model.pkl` produced by the notebook. The notebook is the recommended entry point because it performs the training and evaluation first.
