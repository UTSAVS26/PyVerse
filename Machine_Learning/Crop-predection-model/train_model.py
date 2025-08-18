import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

                                # Load dataset
file_path = "Cat_Crop.csv"
df = pd.read_csv(file_path)


le_soil = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df['SoilType'] = le_soil.fit_transform(df['Soil type'])
df['Season'] = le_season.fit_transform(df['Season'])
df['Crop'] = le_crop.fit_transform(df['Crop'])

X = df[['Rainfall', 'Temperature', 'SoilType', 'Season']]
y = df['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "crop_model.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_season, "season_encoder.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")

print("Model trained and saved!")
