import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load the saved models
svm_model = joblib.load('model_svm.sav')
rf_model = joblib.load('model_rf.sav')
lr_model = joblib.load('model_lr.sav')
nb_model = joblib.load('model_nb.sav')
dt_model = joblib.load('model_dt.sav')

# Define the Streamlit app
def app():
    # Set the title and subtitle of the app
    st.title('COVID Classifier App')
    st.subheader('Enter the input parameters to make a prediction')
    
    # Add a drop down menu to choose between slider and text input
    input_type = option_menu('Select Input Type', ('Slider', 'Text Input'))

    if input_type == 'Slider':
        # Create input widgets for the user to enter data using sliders
        ct_n = st.slider('CtN', 0.0, 50.0, 25.0)
        ct_e = st.slider('CtE', 0.0, 50.0, 25.0)
        ct_rdrp = st.slider('CtRdRp', 0.0, 50.0, 25.0)

        # Create a dictionary to hold the user input
        input_data = {'CtN': ct_n, 'CtE': ct_e, 'CtRdRp': ct_rdrp}

    else:
        # Create input widgets for the user to enter data using text inputs
        ct_n_input = st.text_input("Enter CtN (float value)", 25.0)
        ct_e_input = st.text_input("Enter CtE (float value)", 25.0)
        ct_rdrp_input = st.text_input("Enter CtRdRp (float value)", 25.0)

        # Create a dictionary to hold the user input
        input_data = {'CtN': float(ct_n_input) if ct_n_input else None,
                      'CtE': float(ct_e_input) if ct_e_input else None,
                      'CtRdRp': float(ct_rdrp_input) if ct_rdrp_input else None}

    # Create a menu to select the classifier
    classifier = option_menu('Select a classifier', ('SVM', 'Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Tree'))

    # Create a button to trigger the predictions
    if st.button('Predict'):
        # Convert the dictionary to a Pandas DataFrame
        input_df = pd.DataFrame.from_dict([input_data])

        # Make predictions using the selected classifier
        if classifier == 'SVM':
            prediction = svm_model.predict(input_df)[0]
        elif classifier == 'Random Forest':
            prediction = rf_model.predict(input_df)[0]
        elif classifier == 'Logistic Regression':
            prediction = lr_model.predict(input_df)[0]
        elif classifier == 'Naive Bayes':
            prediction = nb_model.predict(input_df)[0]
        else:
            prediction = dt_model.predict(input_df)[0]

        # Display the predicted class to the user
        st.subheader('Prediction')
        if prediction == 1:
            st.write('Positive')
        else:
            st.write('Negative')
    
# Run the app
if __name__ == '__main__':
    app()
