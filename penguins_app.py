# TODO: Import libraries 
import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier 



# TODO: Display the Penguin Prediction App header and short description
st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library]      
(https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

# TODO: Display the User Input Features sidebar and short description
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)                    
""")

# TODO: Collect the user input features into dataframe 
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# The uploaded_file exist
if uploaded_file is not None: 
    input_df = pd.read_csv(uploaded_file)
    
# Get input_df data from the user input features 
else: 
    def user_input_features(): 
        # Create a list of sidebar for the user input features 
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        
        data = {
            'island':island,
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm':flipper_length_mm,
            'body_mass_g':body_mass_g,
            'sex':sex
        }
        
        features = pd.DataFrame(data, index=[0])
        
        return features 
    
    input_df = user_input_features()
        
# TODO: Combine the user input features with the entire penguins dataset  
penguins_raw = pd.read_csv('penguins_cleaned.csv')

# remove the 'species' column
penguins = penguins_raw.drop(columns=['species'])

# Make df contains the input_df and all data from penguins  
df = pd.concat([input_df, penguins], axis=0)


# TODO: Encoding the ordinal features 
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
    
df = df[:1]    # Selects only the first row (the user input data)

# TODO: Display the user input features
st.subheader('User Input Features')

# Data is from the uploaded_file 
if uploaded_file is not None: 
    st.write(df)
else:    # df data is from the user input features 
    st.write('Awaiting CSV file to be uploaded. Currently using the example input parameters (shown below).')
    st.write(df)

# TODO: Read the 'penguins_clf.pkl into load_clf 
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# TODO: Apply the model to make predictions 
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# TODO: Display the prediction 
st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

# TODO: Display the prediction probability 
st.subheader('Prediction Probability')
st.write(prediction_proba)

