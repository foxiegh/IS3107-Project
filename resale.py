import joblib
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import requests
from warnings import simplefilter
from io import BytesIO
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class NumericalPredictionResale(nn.Module):
    def __init__(self, a=64, a2=32, b=128, c=16):
        super(NumericalPredictionResale, self).__init__()
        self.num_layers = nn.Sequential(
            nn.Linear(9, a2),
            nn.ReLU(),
            # nn.Linear(a, a2),
            # nn.ReLU(),
            )
        
        self.binary_layers = nn.Sequential(
            nn.Linear(190, c),
            nn.ReLU(),
            # nn.Linear(b, c),
            # nn.ReLU()
            )
        self.output_layer = nn.Linear(a2+c, 1)
        
    def forward(self, numerical_inputs, binary_inputs):
        numerical_outputs = self.num_layers(numerical_inputs)
        binary_outputs = self.binary_layers(binary_inputs)
        concatenated_outputs = torch.cat((numerical_outputs, binary_outputs), dim=1)
        final_outputs = self.output_layer(concatenated_outputs)
        return final_outputs

file_dir = r"https://raw.githubusercontent.com/foxiegh/IS3107-Project/main/ML-prediction" # Define file directory
resale_num_cols = ['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 
                'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017',
                "number_sold_around_same_mrt_for_month"]
resale_normalizer = dict()

for i in resale_num_cols:
    norm = joblib.load(BytesIO(requests.get(file_dir + '/' + i + '-resale-normalizer-scaler.save').content)) 
    resale_normalizer[file_dir + '/' + i + '-resale-normalizer-scaler.save'] = norm

resale_model = NumericalPredictionResale(a2=32, c=16)
resale_model.load_state_dict(torch.load(file_dir + '/resale-a2-32-c-16.pt'))

f = open(file_dir+"/resale_cols.txt", "r")
resale_cols_in_order = json.loads(f.read())["columns"]

def predict_resale(town, flat_type, flat_model, 
                         floor_area_sqm, remaining_lease, storey_range, latitude, longitude, 
                          nearest_mrt, distance_to_nearest_mrt, resale_days_since_2017,
                         number_sold_around_same_mrt_for_month, normalizer = resale_normalizer, model = resale_model, cols_in_order = resale_cols_in_order, file_dir=file_dir, multi_pred=False,):

    num_cols = ['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 
                'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017',
                "number_sold_around_same_mrt_for_month"]

    if multi_pred:
        df = pd.DataFrame({'town':town, 'flat_type':flat_type, 'floor_area_sqm':floor_area_sqm,'remaining_lease':remaining_lease,
                        'storey_range':storey_range, 'latitude':latitude, 'longitude':longitude, 
                        'distance_to_nearest_mrt':distance_to_nearest_mrt, 'resale_days_since_2017':resale_days_since_2017, 
                        "flat_model":flat_model, "nearest_mrt":nearest_mrt, 
                        "number_sold_around_same_mrt_for_month":number_sold_around_same_mrt_for_month})
    else:
        df = pd.DataFrame({'town':town, 'flat_type':flat_type, 'floor_area_sqm':floor_area_sqm,'remaining_lease':remaining_lease,
                        'storey_range':storey_range, 'latitude':latitude, 'longitude':longitude, 
                        'distance_to_nearest_mrt':distance_to_nearest_mrt, 'resale_days_since_2017':resale_days_since_2017, 
                        "flat_model":flat_model, "nearest_mrt":nearest_mrt, 
                        "number_sold_around_same_mrt_for_month":number_sold_around_same_mrt_for_month}, index=[0])

    for category in ["town", "flat_model", "nearest_mrt"]:
        encoded = pd.get_dummies(df[[category]])
        df = pd.concat([df.drop([category], axis=1),encoded], axis=1)
        del encoded

    for col in cols_in_order:
        if col not in df:
            df[col] = 0

    for i in num_cols:
        norm = normalizer[file_dir + '/' + i + '-resale-normalizer-scaler.save'] 
        # transform the training data column
        df[i] = norm.transform(df[[i]])

    df = df[cols_in_order]
    
    con_X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    con_X_bin = torch.tensor(df.drop(num_cols, axis=1).values, dtype=torch.float32)


    pred = model(con_X_num, con_X_bin).detach().numpy()
    return pd.DataFrame(pred.reshape([-1,1]), columns=["resale_price"])

print(predict_resale(town="ANG MO KIO", flat_type=2, flat_model="Improved", 
               floor_area_sqm=44.0, remaining_lease=736, storey_range=4, 
               latitude=1.362005, longitude=103.853882, nearest_mrt="NS16",
               distance_to_nearest_mrt=0.999941, resale_days_since_2017=0, 
               number_sold_around_same_mrt_for_month=34, normalizer=resale_normalizer, model=resale_model, file_dir=file_dir, cols_in_order=resale_cols_in_order))