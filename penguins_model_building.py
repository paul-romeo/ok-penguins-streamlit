# Objectives: 
# 1. Convert sex and island data in the 'penguins_cleaned.csv' into the corresponding numerical values
# 2. Output the modified penguins data to 'penguins_clf.pkl'

# TODO: Import library 
import pandas as pd 

# TODO: Read data into penguins 
penguins = pd.read_csv('penguins_cleaned.csv')


# TODO: Process data 
# Ordinal feature encoding 
# https://www.kaggle.com/pratik120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
df 
target = 'species'     # output 

# inputs 
encode = ['sex', 'island']

for col in encode: 
    # Convert:
    # 1. the sex type (male, female) into numerical binary value for sex_male and sex_female to every record in dummy for the encode = sex 
    # 2. the island name into numerical value in island_Biscoe, island_Dream, and island_Torgersen to every record in dummy for encode = island 
    dummy = pd.get_dummies(df[col], prefix=col)     
    df = pd.concat([df,dummy], axis=1)             # append the dummy data into df 
    del df[col]     # delete the specific encode (sex or island) column after appending to the dummy data 
    
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

# Replace the species with its corresponding value for every record in the df 
df['species'] = df['species'].apply(target_encode)


# TODO: Separating X and Y 
X = df.drop('species', axis=1)      # X contains df data without species 
Y = df['species']                   # Y contains only species data in numerical value 

# TODO: Build random forest model 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# TODO: Saving the model 
import pickle 
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))

