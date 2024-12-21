import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv(r'C:\Users\varun\Downloads\Summer Intern Project\Plant_Parameters.csv')

# Split the data into features (x) and target variable (y)
x = df[['pH', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature']].values
y = df['Plant Type'].values

# Standardize the features
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Train the Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_scaled, y)

def predict_crop(input_values, scaler, model):
    # Create a DataFrame with user input
    input_df = pd.DataFrame([input_values], columns=['pH', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature'])

    # Check and fill missing values for each column separately with the mean
    for col in input_df.columns:
        if pd.isnull(input_df[col].values[0]):
            input_df[col].fillna(df[col].mean(), inplace=True)

    # Standardize the input using the same scaler used for training
    input_scaled = scaler.transform(input_df.values)

    # Make the prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit web app
def main():
    st.title("Crop Prediction App")

    # Input form for user
    st.sidebar.header("User Input")

    # Input fields for pH, Phosphorus, Potassium, Moisture, and Temperature
    pH = st.sidebar.number_input("Enter pH", min_value=df['pH'].min(), max_value=df['pH'].max(), step=0.1)
    phosphorus = st.sidebar.number_input("Enter Phosphorus", min_value=df['Phosphorus'].min(), max_value=df['Phosphorus'].max(), step=0.1)
    potassium = st.sidebar.number_input("Enter Potassium", min_value=df['Potassium'].min(), max_value=df['Potassium'].max(), step=0.1)
    moisture = st.sidebar.number_input("Enter Moisture", min_value=df['Moisture'].min(), max_value=df['Moisture'].max(), step=0.1)
    temperature = st.sidebar.number_input("Enter Temperature", min_value=df['Temperature'].min(), max_value=df['Temperature'].max(), step=0.1)

    # Make prediction
    if st.sidebar.button("Predict"):
        user_input = [pH, phosphorus, potassium, moisture, temperature]
        predicted_crop = predict_crop(user_input, sc, classifier)
        st.success(f"Predicted Crop: {predicted_crop}")

if __name__ == "__main__":
    main()
