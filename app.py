import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('models/logistic_regression_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Streamlit App UI
st.title('Text Classification App')
st.write("Enter text below to predict its category.")

# Text input
text = st.text_area('Text to Classify', '')

# Prediction function
if st.button('Predict'):
    if text:
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        st.write(f'Predicted Category: {prediction[0]}')
    else:
        st.write("Please enter some text to classify.")
