import joblib
import pandas as pd

def test_prediction():
    model = joblib.load('model/house_price_model.pkl')
    sample_data = {
        'Feature1': 3,
        'Feature2': 1200,
        'Feature3': 2,
        'Feature4': 1
    }
    input_df = pd.DataFrame([sample_data])
    prediction = model.predict(input_df)
    assert prediction[0] > 0, "Prediction should be greater than 0"

if __name__ == "__main__":
    test_prediction()
    print("Test passed!")
