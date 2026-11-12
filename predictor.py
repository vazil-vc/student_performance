import pandas as pd
import joblib


model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")


def predict_score(input_data):


    input_df = pd.DataFrame([input_data])


    input_df = pd.get_dummies(
        input_df,
        columns=[
            "Teacher_Quality",
            "Family_Income",
            "Parental_Involvement",
            "Access_to_Resources",
            "Motivation_Level",
            "Peer_Influence",
            "Parental_Education_Level",
            "Distance_from_Home"
        ],
        drop_first=True
    )
    input_df["Extracurricular_Activities"] = (
    input_df["Extracurricular_Activities"].map({"Yes": 1, "No": 0})
    )

    input_df["Internet_Access"] = (
    input_df["Internet_Access"].map({"Yes": 1, "No": 0})
    )

    input_df["Learning_Disabilities"] = (
    input_df["Learning_Disabilities"].map({"Yes": 1, "No": 0})
    )

    input_df["Gender"] = (
    input_df["Gender"].map({"Male": 1, "Female": 0})
    )

    input_df["School_Type"] = (
    input_df["School_Type"].map({"Public": 1, "Private": 0})
    )

  
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    print(input_df.dtypes)
    print(input_df.head())

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return prediction[0]