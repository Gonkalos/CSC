import numpy as np
import pandas as pd

# Read dataset
df = pd.read_csv('flights_dataset.csv')

# Dataset info
print(df.info())
print('--------')

# Drop unwanted columns
df.drop(['hour', 'minute', 'tailnum'], 1, inplace=True)

# Infer objects type
df.infer_objects()

# Check and replace missing with -99 (masking)
print(df.isnull().sum())
print('--------')
df.fillna(-99, inplace=True)

# Frequency distribution of categories within a feature
print(df['dest'].unique())
print('Unique count: %d' %df['dest'].value_counts().count())
print('--------')
print(df['dest'].value_counts())
print('--------')

# Function to encode all non-(int/float) features in a dataframe
def label_encoding(df):
    
    label_dictionary = {}

    # Iterate over non-(int/float) features
    for feature in df.select_dtypes(exclude=['int64', 'float64']):

        # Get unique values in feature
        uniqueValues = df[feature].unique()
        uniqueCount = len(uniqueValues)
        
        # Store the relation between the label and the integer that encodes it
        relation = {}
        for i in range(0, uniqueCount):
            relation[uniqueValues[i]] = i
            label_dictionary[feature] = relation
        
        # Replace each value with the integer that encodes it
        for value in uniqueValues:
            encode = label_dictionary[feature].get(value)
            df.loc[(df[feature] == value), feature] = encode
        
    return df, label_dictionary

# Function to decode what was previously encoded 
def label_decoding(df_labelled, label_dictionary):
    
    # Iterate over encoded features
    features = list(label_dictionary.keys())
    for feature in features:

        # Iterate over unique values
        values = list(label_dictionary[feature].keys())
        for value in values:

            # Replace each integer with value that decodes it
            encode = label_dictionary[feature][value]
            df_labelled.loc[(df_labelled[feature] == encode), feature] = value

    return df_labelled

df_labelled, label_dictionary = label_encoding(df)
print(df_labelled['dest'].unique())
print('Unique count after Label Encoding: %d' %df_labelled['dest'].value_counts().count())
df_labelled_decoded = label_decoding(df_labelled, label_dictionary)
print(df_labelled_decoded['dest'].unique())
print('Unique count after dec.: %d' %df_labelled_decoded['dest'].value_counts().count())
print('--------')

# Apply one-hot encoding to the origin column
print('Unique Origin:', df['origin'].unique())
print(df.columns.values)
df_pandas_ohe = pd.get_dummies(df['origin'], prefix='origin')
print(df_pandas_ohe.head())