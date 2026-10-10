"""
Gradio frontend for the Colombo land-price regression activity.

Run after app.ipynb has trained and exported model.pkl:
    python app.py

The model predicts advertised price per perch, not a verified transaction price.
"""
from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = Path(__file__).with_name("model.pkl")

# These must match the feature columns and preprocessing pipeline used in app.ipynb.
FEATURE_COLUMNS = [
    "Address", "Land_size(Perches)", "Land_type", "Distance from fort",
    "count_govtschools_A", "min_dist_govtschools_a",
    "count_govtschools_B", "min_dist_govtschools_b",
    "count_semigovtschools", "min_dist_semigovtschools",
    "count_intlschools", "min_dist_intlschools",
    "count_uni", "min_dist_uni", "min_dist_nearest_express",
    "min_dist_nearest_railway", "min_dist_nearest_bank",
    "count_banks_within_2km", "min_dist_nearest_FinanceCompany",
    "count_FinanceCompanies_within_2km", "min_dist_nearest_Govt_Hospital",
    "count_Govt_Hospitals", "min_dist_nearest_Pvt_Hospital",
    "count_Pvt_Hospital", "min_dist_nearest_Pvt_Med_center",
    "count_Pvt_Med_Centers", "min_dist_nearest_Supermarket",
    "count_Supermarkets_within2km", "min_dist_nearest_Fuel_station",
    "count_Fuel_Stations_within2km",
]

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("Address", "Land_type")]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Run app.ipynb first to train and export the model."
        )
    return joblib.load(MODEL_PATH)

def predict_one(*values):
    model = load_model()
    row = dict(zip(FEATURE_COLUMNS, values))
    sample = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(sample)[0])
    return f"Estimated advertised price per perch: LKR {prediction:,.0f}"

def predict_csv(file_obj):
    if file_obj is None:
        return None, "Please upload a CSV file."
    data = pd.read_csv(file_obj.name)
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        return None, "Missing required feature columns: " + ", ".join(missing)
    model = load_model()
    result = data.copy()
    result["Predicted_Price_per_Perch_Rs"] = model.predict(result[FEATURE_COLUMNS])
    return result, f"Predictions generated for {len(result):,} rows."

with gr.Blocks(title="Colombo Land Price Estimator") as demo:
    gr.Markdown(
        "# Colombo District Land Price Estimator\n"
        "AML Part 1 — Random Forest vs XGBoost regression.\n\n"
        "**Important:** These are model estimates of advertised price per perch, "
        "not verified property valuations."
    )
    with gr.Tab("Predict one property"):
        gr.Markdown("Enter the property/location features. Distances use the units represented in the original dataset.")
        inputs = []
        for column in FEATURE_COLUMNS:
            if column == "Address":
                component = gr.Textbox(label=column, placeholder="e.g. battaramulla")
            elif column == "Land_type":
                component = gr.Textbox(label=column, placeholder="e.g. Residential")
            else:
                component = gr.Number(label=column, value=0)
            inputs.append(component)
        predict_button = gr.Button("Predict price per perch", variant="primary")
        single_output = gr.Textbox(label="Prediction")
        predict_button.click(predict_one, inputs=inputs, outputs=single_output)

    with gr.Tab("Predict from CSV"):
        gr.Markdown(
            "Upload a CSV containing every feature column listed in the notebook. "
            "Do not include the target or advertised price columns as model inputs."
        )
        csv_input = gr.File(label="Feature CSV", file_types=[".csv"])
        csv_button = gr.Button("Generate CSV predictions", variant="primary")
        csv_output = gr.Dataframe(label="Predictions")
        csv_status = gr.Textbox(label="Status")
        csv_button.click(predict_csv, inputs=csv_input, outputs=[csv_output, csv_status])

if __name__ == "__main__":
    demo.launch()
