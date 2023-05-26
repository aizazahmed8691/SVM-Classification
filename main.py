import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from flask import Flask, request
import pymongo
from bson import ObjectId
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Connect to MongoDB and retrieve data
client = pymongo.MongoClient("mongodb+srv://Usman123:rqui3RT5nZBaBE7L@cluster0.cnukbzb.mongodb.net/?retryWrites=true&w=majority")
db = client["test"]
collection = db["values"]
data = pd.DataFrame(list(collection.find()))

# Preprocess data
# scaler = StandardScaler()
# numerical_cols = ['Acc X', 'Acc Y', 'Acc Z']
# data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save the _id field in a separate column
data['_id_str'] = data['_id'].astype(str)
# Save the _id field in a separate column
data['_Lat_str'] = data['Latitude'].astype(str)
# Save the _id field in a separate column
data['_Long_str'] = data['Longitude'].astype(str)
# save the Acc X field in a separate column
data['Acc_X_str'] = data['Acc X'].astype(str)
# save the Acc Y field in a separate column
data['Acc_Y_str'] = data['Acc Y'].astype(str)
# save the Acc Z field in a separate column
data['Acc_Z_str'] = data['Acc Z'].astype(str)


# Drop unnecessary columns
data = data.drop(['_id','Latitude','Longitude'], axis=1)

# Set instance ID and index
# instance_id = "my_instance_id"
index = 0

# Load trained model
svm = load('svm_model.joblib')

# Create a new DataFrame
mapvalues_data = pd.DataFrame(columns=['_id', 'Longitude', 'Latitude', 'Acc X', 'Acc Y', 'Acc Z', 'Class'])

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    # Extract features from row
    X = row[['Acc X', 'Acc Y', 'Acc Z']].values.reshape(1, -1)

    # Use the trained SVM model to predict the class for the row
    y_pred = svm.predict(X)[0]

    # Get the _id value for the row
    id_value = row['_id_str']

    # Get the Lat value for the row
    Lat_value = row['_Lat_str']

    # Get the Long value for the row
    Long_value = row['_Long_str']

    Acc_X = row['Acc_X_str']
    Acc_Y = row['Acc_Y_str']
    Acc_Z = row['Acc_Z_str']

    # Create a dictionary of the new row data
    new_row = {
        '_id': ObjectId(id_value),
        'Longitude': Long_value,
        'Latitude': Lat_value,
        'Acc X': Acc_X,
        'Acc Y': Acc_Y,
        'Acc Z': Acc_Z,
        'Class': y_pred
    }

     # Add the new row to the DataFrame using the loc[] accessor
    mapvalues_data.loc[len(mapvalues_data)] = new_row

    print(mapvalues_data)

# # Save the new DataFrame to MongoDB
# mapvalues_data_dict = mapvalues_data.to_dict('records')
# collection_mapvalues = db["classified"]
# collection_mapvalues.insert_many(mapvalues_data_dict)
mapvalues_data_dict = mapvalues_data.to_dict('records')
collection_mapvalues = db["classified"]

for record in mapvalues_data_dict:
    id_value = record["_id"]
    collection_mapvalues.update_one({"_id": id_value}, {"$set": record}, upsert=True)


@app.route('/')
def home():
    return 'Flask server'

if __name__ == '__main__':
    app.run()

