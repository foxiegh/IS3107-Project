import joblib
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class NumericalPredictionConstruction(nn.Module):
    def __init__(self, a=64, a2=32, b=128, c=64):
        super(NumericalPredictionConstruction, self).__init__()
        self.num_layers = nn.Sequential(
            nn.Linear(8, a2),
            nn.ReLU(),
            # nn.Linear(a, a2),
            # nn.ReLU(),
            )
        
        self.binary_layers = nn.Sequential(
            nn.Linear(165, c),
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

file_dir = f"https://raw.githubusercontent.com/foxiegh/IS3107-Project/main/ML-prediction" # Define file directory
construction_num_cols = ['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 
                'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017']
                
construction_normalizer = dict()

for i in construction_num_cols:
    norm = joblib.load(file_dir + '/' + i + '-construction-normalizer-scaler.save') 
    construction_normalizer[file_dir + '/' + i + '-construction-normalizer-scaler.save'] = norm

construction_model = NumericalPredictionConstruction(a2=32, c=16)
construction_model.load_state_dict(torch.load(file_dir + '/construction-a2-32-c-16.pt'))

f = open(file_dir+"/construction_cols.txt", "r")
construction_cols_in_order = json.loads(f.read())["columns"]


def predict_construction(floor_area_sqm, remaining_lease, distance_to_nearest_mrt, 
                         resale_days_since_2017, flat_type, storey_range, latitude, 
                         longitude, flat_model, nearest_mrt, normalizer = construction_normalizer, model = construction_model, cols_in_order = construction_cols_in_order, file_dir=file_dir, multi_pred=False, ):


    num_cols = ['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017',]

    # flat_type	floor_area_sqm	remaining_lease	storey_range	latitude	longitude	distance_to_nearest_mrt	resale_days_since_2017
    
    if multi_pred:
        df = pd.DataFrame({'flat_type':flat_type, 'floor_area_sqm':floor_area_sqm,'remaining_lease':remaining_lease,'storey_range':storey_range, 'latitude':latitude, 'longitude':longitude, 'distance_to_nearest_mrt':distance_to_nearest_mrt, 'resale_days_since_2017':resale_days_since_2017, 
                    "flat_model":flat_model, "nearest_mrt":nearest_mrt})
    else:
        df = pd.DataFrame({'flat_type':flat_type, 'floor_area_sqm':floor_area_sqm,'remaining_lease':remaining_lease,'storey_range':storey_range, 'latitude':latitude, 'longitude':longitude, 'distance_to_nearest_mrt':distance_to_nearest_mrt, 'resale_days_since_2017':resale_days_since_2017, 
                    "flat_model":flat_model, "nearest_mrt":nearest_mrt}, index=[0])

    # print("1.---------------------------------------\n",df.loc[:, (df[0:1] != 0).any()][0:1])

    for category in ["flat_model", "nearest_mrt"]:
        encoded = pd.get_dummies(df[[category]])
        df = pd.concat([df.drop([category], axis=1),encoded], axis=1)
        del encoded

    for col in cols_in_order:
        if col not in df:
            df[col] = 0

    # print("3.---------------------------------------\n",df.loc[:, (df[0:1] != 0).any()][0:1])

    for i in num_cols:
        norm = normalizer[file_dir + '/' + i + '-construction-normalizer-scaler.save'] 
        # transform the training data column
        df[i] = norm.transform(df[[i]])

    df = df[cols_in_order]

    # print("4.---------------------------------------\n",df.loc[:, (df[0:1] != 0).any()][0:1])

    con_X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    con_X_bin = torch.tensor(df.drop(num_cols, axis=1).values, dtype=torch.float32)

    # print(con_X_num.shape)
    # print(con_X_bin.shape)

    model = NumericalPredictionConstruction(a2=32, c=16)
    model.load_state_dict(torch.load(file_dir+'/construction-a2-32-c-16.pt'))
    pred = model(con_X_num, con_X_bin).detach().numpy()
    return pd.DataFrame(pred.reshape([-1,1]), columns=["resale_price"])

predict_construction(flat_type=2, flat_model="Improved", 
        floor_area_sqm=44.0, remaining_lease=736, storey_range=4, 
        latitude=1.362005, longitude=103.853882, nearest_mrt="NS16",
        distance_to_nearest_mrt=0.999941, resale_days_since_2017=0, normalizer=construction_normalizer, 
        model=construction_model, cols_in_order=construction_cols_in_order, file_dir=file_dir, multi_pred=False)

# predict_construction(flat_type=3, flat_model="New Generation", 
#                         floor_area_sqm=68.0, remaining_lease=744, storey_range=2, 
#                         latitude=1.366201, longitude=103.857201, nearest_mrt="TE6",
#                         distance_to_nearest_mrt=0.945375, resale_days_since_2017=0)

# predict_construction(flat_type=6, flat_model="Maisonette", 
#                         floor_area_sqm=146.0, remaining_lease=761, storey_range=4, 
#                         latitude=1.420500	, longitude=103.832375, nearest_mrt="NS14",
#                         distance_to_nearest_mrt=0.351151, resale_days_since_2017=2250)

# predict_construction(flat_type=6, flat_model="Apartment", 
#                         floor_area_sqm=164.0, remaining_lease=819, storey_range=1, 
#                         latitude=1.421062, longitude=103.838806, nearest_mrt="NS14",
#                         distance_to_nearest_mrt=0.765432, resale_days_since_2017=2250)


# mse = mean_squared_error(df[["resale_price"]],pred)
# print("mse: {}, mae: {}".format(mse, np.sqrt(mse)))