import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

CSV_COL_FC12 = "FC1,2"
CSV_COL_FC21 = "FC2,1"
CSV_COL_FC22 = "FC2,2"
CSV_COL_FC23 = "FC2,3"
CSV_COL_FC32 = "FC3,2"
FEAT_FC21_DIV_FC22 = "FC2,1/FC2,2"
FEAT_FC23_DIV_FC22 = "FC2,3/FC2,2"
FEAT_FC12_DIV_FC22 = "FC1,2/FC2,2"
FEAT_FC32_DIV_FC22 = "FC3,2/FC2,2"
REQUIRED_FC_COLUMNS = [CSV_COL_FC12, CSV_COL_FC21, CSV_COL_FC22, CSV_COL_FC23, CSV_COL_FC32]


class GradientBoostingPitCorrosionPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.targets = None
        
    def load_model(self, model_name='gradient_boosting_model'):
        self.model = joblib.load(f'{model_name}.joblib')
        self.features = joblib.load('gradient_boosting_features.joblib')
        self.targets = joblib.load('gradient_boosting_targets.joblib')
        print("Model loaded successfully")

    def load_input_data(self, file_path):
        print(f"Reading input file: {file_path}")
        data = pd.read_csv(file_path)
        if data.columns[0].startswith("Unnamed") or data.columns[0] == "":
            data = data.rename(columns={data.columns[0]: "Sample_ID"})
        data[REQUIRED_FC_COLUMNS] = data[REQUIRED_FC_COLUMNS].round(2)
        return data
    
    def preprocess_data(self, data):
        processed_data = pd.DataFrame()
        processed_data[FEAT_FC21_DIV_FC22] = data[CSV_COL_FC21] / data[CSV_COL_FC22]
        processed_data[FEAT_FC23_DIV_FC22] = data[CSV_COL_FC23] / data[CSV_COL_FC22]
        processed_data[FEAT_FC12_DIV_FC22] = data[CSV_COL_FC12] / data[CSV_COL_FC22]
        processed_data[FEAT_FC32_DIV_FC22] = data[CSV_COL_FC32] / data[CSV_COL_FC22]
        processed_data = processed_data[self.features]
        return processed_data
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions


def main():
    predictor = GradientBoostingPitCorrosionPredictor()
    predictor.load_model()
    input_file = 'data\\Experimental_data.csv'
    input_data = predictor.load_input_data(input_file)
    print("\n=== Input Data (First 5 Rows) ===")
    print(input_data.head().to_string(index=False))

    X = predictor.preprocess_data(input_data)
    predictions = predictor.predict(X)

    if "No." in input_data.columns:
        pit_ids = input_data["No."].values
    elif "Sample_ID" in input_data.columns:
        pit_ids = input_data["Sample_ID"].values
    else:
        pit_ids = input_data.iloc[:, 0].values

    x_col, y_col = predictor.targets[0], predictor.targets[1]
    results_df = pd.DataFrame({
        "No.": pit_ids,
        x_col: np.round(predictions[:, 0], 5),
        y_col: np.round(predictions[:, 1], 5),
    })

    output_file = 'prediction_location_results.csv'
    print(f"\nSaving prediction results to: {output_file}")
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
