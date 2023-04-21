import json
from datetime import datetime
import requests
import pandas as pd
import json
from geopy.distance import geodesic as GD
from sqlalchemy import create_engine
from sqlalchemy import inspect

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import urllib
import torch
import torch.nn as nn
import geopy.distance
# from math import dist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from io import StringIO
from pandas import json_normalize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

default_args = {
    'owner': 'airflow',
}

with DAG(
    'Project_Training',
    default_args=default_args,
    description='HDB Data tranformation',
    #Can be set to daily to retrieve new data everyday
    schedule_interval="@weekly",
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    
    def getEngine():
        # engine = create_engine("mysql+mysqldb://wsl_root:password@192.168.192.1:3306/hdb")
        engine = create_engine("mysql+mysqldb://wsl_root:password@172.20.224.1:3306/hdb")

        return engine

    def preprocess_df(df):
        df["remaining_lease"] = df["remaining_lease"].str.findall(r'\d+ years').str[0].str.findall(r'\d+').str[0].fillna(0).astype(int) * 12 \
                + df["remaining_lease"].str.findall(r'\d+ months').str[0].str.findall(r'\d+').str[0].fillna(0).astype(int)
        df["distance_to_nearest_mrt"] = df["distance_to_nearest_mrt"].str.findall(r'\d+.\d+').str[0].fillna(0).astype(np.float32)
        df["latitude"] = df["latitude"].astype(np.float32)
        df["longitude"] = df["longitude"].astype(np.float32)
        df["resale_days_since_2017"] = (pd.to_datetime(df["month"], format="%Y-%m") - pd.to_datetime("2017-01-01")).dt.days
        df = df.drop(["month"], axis=1)

        aggregate_result=pd.DataFrame({'number_sold_around_same_mrt_for_month':df.groupby(["nearest_mrt","resale_days_since_2017"])[['nearest_mrt','resale_days_since_2017']].size()}).reset_index()
        df = df.merge(aggregate_result, on=["nearest_mrt","resale_days_since_2017"])

        # Label encode df
        df.loc[df["storey_range"]=='01 TO 03', "storey_range"] = 1
        df.loc[df["storey_range"]=='04 TO 06', "storey_range"] = 2
        df.loc[df["storey_range"]=='07 TO 09', "storey_range"] = 3
        df.loc[df["storey_range"]=='10 TO 12', "storey_range"] = 4
        df.loc[df["storey_range"]=='13 TO 15', "storey_range"] = 5
        df.loc[df["storey_range"]=='16 TO 18', "storey_range"] = 6
        df.loc[df["storey_range"]=='19 TO 21', "storey_range"] = 7
        df.loc[df["storey_range"]=='22 TO 24', "storey_range"] = 8
        df.loc[df["storey_range"]=='25 TO 27', "storey_range"] = 9
        df.loc[df["storey_range"]=='28 TO 30', "storey_range"] = 10
        df.loc[df["storey_range"]=='31 TO 33', "storey_range"] = 11
        df.loc[df["storey_range"]=='34 TO 36', "storey_range"] = 12
        df.loc[df["storey_range"]=='37 TO 39', "storey_range"] = 13
        df.loc[df["storey_range"]=='40 TO 42', "storey_range"] = 14
        df.loc[df["storey_range"]=='43 TO 45', "storey_range"] = 15
        df.loc[df["storey_range"]=='46 TO 48', "storey_range"] = 16
        df.loc[df["storey_range"]=='49 TO 51', "storey_range"] = 17
        df.loc[:,"storey_range"] = df[["storey_range"]].astype(int)

        df.loc[df["flat_type"]=='1 ROOM', "flat_type"] = 1
        df.loc[df["flat_type"]=='2 ROOM', "flat_type"] = 2
        df.loc[df["flat_type"]=='3 ROOM', "flat_type"] = 3
        df.loc[df["flat_type"]=='4 ROOM', "flat_type"] = 4
        df.loc[df["flat_type"]=='5 ROOM', "flat_type"] = 5
        df.loc[df["flat_type"]=='EXECUTIVE', "flat_type"] = 6
        df.loc[df["flat_type"]=='MULTI-GENERATION', "flat_type"] = 7
        df.loc[:,"flat_type"] = df[["flat_type"]].astype(int)
        return df

    def encode(**kwargs):
        ti = kwargs['ti']
        df = preprocess_df(pd.read_sql_query('''SELECT * FROM closest_mrt''', con=getEngine()))

        y = df[["resale_price"]]

        construction_X = df[["flat_type", "flat_model", "floor_area_sqm", "remaining_lease", 
                    "storey_range", "latitude", "longitude", "nearest_mrt", "distance_to_nearest_mrt", "resale_days_since_2017"]].copy()

        resale_X = df[["town", "flat_type", "flat_model", "floor_area_sqm", "remaining_lease", 
                    "storey_range", "latitude", "longitude", "nearest_mrt", "distance_to_nearest_mrt", "resale_days_since_2017", "number_sold_around_same_mrt_for_month"]].copy()

        construction_X.to_sql("construction_x", getEngine(), if_exists = 'replace', index=False)
        resale_X.to_sql("resale_x", getEngine(), if_exists = 'replace', index=False)
        y.to_sql("y", getEngine(), if_exists = 'replace', index=False)
        # ti.xcom_push('construction_X', construction_X)
        # ti.xcom_push('resale_X', resale_X)
        # ti.xcom.push('y', y)

        # Encode construction_X
        for category in ["flat_model", "nearest_mrt"]:
            encoded = pd.get_dummies(construction_X[[category]], drop_first=True)
            construction_X = pd.concat([construction_X.drop([category], axis=1),encoded], axis=1)
            del encoded

        # Encode resale_X
        for category in ["town", "flat_model", "nearest_mrt"]:
            encoded = pd.get_dummies(resale_X[[category]], drop_first=True)
            resale_X = pd.concat([resale_X.drop([category], axis=1),encoded], axis=1)
            del encoded


    # Saves new fitted preprocessing Normalizer
    def split_normalize(X, y, num_cols, test_size=0.2, dataFrame_name="construction"):
        file_dir = "/home/jsoh/IS3107" # Define file directory

        print("y: ", y.dtypes)
        print(y)
        normalization_or_standardize = "normalizer"
        resale_X_train, resale_X_test, resale_y_train, resale_y_test = train_test_split(X, y, test_size=test_size)
        
        
        if normalization_or_standardize == "standardize":
        # numerical features
        # apply standardization on numerical features
            for i in num_cols:
                # fit on training data column
                scale = StandardScaler().fit(resale_X_train[[i]])
                # transform the training data column
                resale_X_train[i] = scale.transform(resale_X_train[[i]])
                # transform the testing data column
                resale_X_test[i] = scale.transform(resale_X_test[[i]])
                
                import joblib
                joblib.dump(scale, file_dir + '/'+ i + '-'+ dataFrame_name +'-standardize-scaler.save') 
                

        elif normalization_or_standardize == "normalizer":
            for i in num_cols:
                # fit on training data column
                norm = MinMaxScaler().fit(resale_X_train[[i]])
                # transform the training data column
                resale_X_train[i] = norm.transform(resale_X_train[[i]])
                # transform the testing data column
                resale_X_test[i] = norm.transform(resale_X_test[[i]])

                
                import joblib
                joblib.dump(norm, file_dir + '/'+ i + '-'+dataFrame_name+'-normalizer-scaler.save') 

        f = open(file_dir+"/"+dataFrame_name+"_cols.txt", "r")
        cols_in_oder = json.loads(f.read())["columns"]
        for col in cols_in_oder:
            if col not in resale_X_train:
                resale_X_train[col] = 0
            if col not in resale_X_test:
                resale_X_test[col] = 0
        resale_X_test = resale_X_test[cols_in_oder]
        resale_X_train = resale_X_train[cols_in_oder]

        train_X_numeric = torch.tensor(resale_X_train[num_cols].values, dtype=torch.float32)
        train_X_binary = torch.tensor(resale_X_train.drop(num_cols, axis=1).values, dtype=torch.float32)
        test_X_numeric = torch.tensor(resale_X_test[num_cols].values, dtype=torch.float32)
        test_X_binary = torch.tensor(resale_X_test.drop(num_cols, axis=1).values, dtype=torch.float32)
        train_y_ = torch.tensor(resale_y_train.values, dtype=torch.float32)
        test_y_ = torch.tensor(resale_y_test.values, dtype=torch.float32)
        return train_X_numeric, train_X_binary, test_X_numeric, test_X_binary, train_y_, test_y_
    


    #Training of model
    def train_model_plot(model, 
                     train_X_numeric, train_X_binary, train_y_, 
                     test_X_numeric, test_X_binary, test_y_,
                     max_epoch = 100, skip_plot = False, batch_size=256, verbose = True, early_stopping = 3):
        time_tester = time.perf_counter()
        loss_over_n = {"Epoch": [], "test_error" : [], "train_error": []}
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        # Train the model
        # TIME_LIMIT = 9999
        stop_counter = 0
        epoch = 1
        end = early_stopping

        if not skip_plot:
            with torch.no_grad():
                    # loss_over_n["Epoch"].append(0)
                    y_pred = model(test_X_numeric, test_X_binary)
                    test_loss = loss_fn(test_y_, y_pred).item()
                    print("Initial test error ", test_loss)
                    # loss_over_n["test_error"].append(test_loss)

                    y_train_pred = model(train_X_numeric, train_X_binary)
                    train_loss = loss_fn(train_y_, y_train_pred).item()
                    print("Initial training error ", train_loss)
                    # loss_over_n["train_error"].append(train_loss)
        # time_limit = time.perf_counter()

        
        prev_loss = 999999999999999999999999999999999999999999
        prev_test_loss = 999999999999999999999999999999999999999999
        while True:
            curr_loss = 0
            
            # if time.perf_counter() - time_limit > TIME_LIMIT:
            #     break
            for i in range(0, len(train_X_numeric), batch_size):
                    optimizer.zero_grad()

                    batch_X_training_dat = train_X_numeric[i:i+batch_size]
                    batch_X_training_img_dat = train_X_binary[i:i+batch_size]
                    batch_y_training_dat = train_y_[i:i+batch_size]

                    outputs = model(batch_X_training_dat, batch_X_training_img_dat)
                    loss = loss_fn(outputs, batch_y_training_dat)
                    loss.backward()
                    curr_loss += loss.item()
                    optimizer.step()
                    # if time.perf_counter() - time_limit > TIME_LIMIT:
                    #     break

            if not skip_plot:
                with torch.no_grad():
                        loss_over_n["Epoch"].append(epoch)
                        y_pred = model(test_X_numeric, test_X_binary)
                        test_loss = loss_fn(test_y_, y_pred).item()
                        loss_over_n["test_error"].append(test_loss)

                        y_train_pred = model(train_X_numeric, train_X_binary)
                        train_loss = loss_fn(train_y_, y_train_pred).item()
                        loss_over_n["train_error"].append(train_loss)
                        if test_loss < prev_test_loss:
                            prev_test_loss = test_loss
                            end = early_stopping
                        else:
                            end -= 1
                            if verbose:
                                print("Test loss not dropping at epoch {}".format(epoch))
                        if end == 0:
                            break
            if verbose and epoch % 250 == 0:
                print("Epoch {}  :\t loss {}".format(epoch, curr_loss))

            stop_counter += 1
            if prev_loss > curr_loss:
                stop_counter = 0
                prev_loss = curr_loss
            epoch += 1
            if epoch >= max_epoch:
                break
            if stop_counter >= 3:
                break
        # Use the model to make predictions on test data
        # y_pred = model(test_X_)
        # print("final error,", mean_squared_error(test_y_, y_pred))

        if not skip_plot:
            # import seaborn as sns
            plt.figure()
            loss_over_n = pd.DataFrame(loss_over_n)
            sns.lineplot(x='Epoch', y='value', hue='variable', 
                        data=pd.melt(loss_over_n, ['Epoch']))
        if verbose:
            print("Time used for model training: {}".format(time.perf_counter() - time_tester))
        return model   

    class NumericalPredictionConstruction(nn.Module):
        def __init__(self, a=64, a2=32, b=128, c=64):
            super(NumericalPredictionConstruction, self).__init__()
            self.num_layers = nn.Sequential(
                nn.Linear(8, a2),
                nn.ReLU(),
                )
            
            self.binary_layers = nn.Sequential(
                nn.Linear(165, c),
                nn.ReLU(),
                )
            self.output_layer = nn.Linear(a2+c, 1)
            
        def forward(self, numerical_inputs, binary_inputs):
            numerical_outputs = self.num_layers(numerical_inputs)
            binary_outputs = self.binary_layers(binary_inputs)
            concatenated_outputs = torch.cat((numerical_outputs, binary_outputs), dim=1)
            final_outputs = self.output_layer(concatenated_outputs)
            return final_outputs
        
    class NumericalPredictionResale(nn.Module):
        def __init__(self, a=64, a2=32, b=128, c=64):
            super(NumericalPredictionResale, self).__init__()
            self.num_layers = nn.Sequential(
                nn.Linear(9, a2),
                nn.ReLU(),
                )
            
            self.binary_layers = nn.Sequential(
                nn.Linear(190, c),
                nn.ReLU(),
                )
            self.output_layer = nn.Linear(a2+c, 1)
            
        def forward(self, numerical_inputs, binary_inputs):
            numerical_outputs = self.num_layers(numerical_inputs)
            binary_outputs = self.binary_layers(binary_inputs)
            concatenated_outputs = torch.cat((numerical_outputs, binary_outputs), dim=1)
            final_outputs = self.output_layer(concatenated_outputs)
            return final_outputs
    
    def hyper_param_resale(**kwargs):
        ti = kwargs['ti']
        hypers = [{"a":0, "a2":32, "b":0, "c":16},]
        epoch = 99999999999999999999999

        # Split train and test
        construction_X = pd.read_sql_query('''SELECT * FROM construction_x''', con=getEngine())
        y = pd.read_sql_query('''SELECT * FROM y''', con=getEngine())
        y["resale_price"] = y['resale_price'].astype(np.float32)
        # construction_X = ti.xcom_pull(task_ids='encode', key='construction_X')
        # y = ti.xcom_pull(task_ids='encode', key='y')

        train_X_numeric, train_X_binary, test_X_numeric, test_X_binary, train_y_, test_y_ = split_normalize(construction_X, y, 
            num_cols=['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017',],
            dataFrame_name="construction")

        for combi in hypers:

            # Train and predict
            model = NumericalPredictionConstruction(**combi)

            model = train_model_plot(model, 
                                train_X_numeric, train_X_binary, train_y_, 
                                test_X_numeric, test_X_binary, test_y_,
                                max_epoch = epoch, skip_plot = False, batch_size=65536, verbose = True)
            
            # Define file directory
            torch.save(model.state_dict(), '/home/jsoh/IS3107/construction-a2-'+str(combi["a2"])+'-c-'+str(combi["c"])+'.pt')

    

    def hyper_param_construction(**kwargs):
        ti = kwargs['ti']
        hypers = [{"a":0, "a2":32, "b":0, "c":16},]

        epoch = 999999999999999999999

        # Split train and test
        resale_X = pd.read_sql_query('''SELECT * FROM resale_x''', con=getEngine())

        y = pd.read_sql_query('''SELECT * FROM y''', con=getEngine())
        y["resale_price"] = y['resale_price'].astype(np.float32)
        # resale_X = ti.xcom_pull(task_ids='encode', key='resale_X')
        # y = ti.xcom_pull(task_ids='encode', key='y')

        train_X_numeric, train_X_binary, test_X_numeric, test_X_binary, train_y_, test_y_ = split_normalize(resale_X, y, 
            num_cols=['flat_type', 'floor_area_sqm','remaining_lease', 'storey_range','latitude', 'longitude', 'distance_to_nearest_mrt', 'resale_days_since_2017',"number_sold_around_same_mrt_for_month"],
            dataFrame_name="resale")


        for combi in hypers:

            # Train and predict
            model = NumericalPredictionResale(**combi)

            model = train_model_plot(model, 
                                train_X_numeric, train_X_binary, train_y_, 
                                test_X_numeric, test_X_binary, test_y_,
                                max_epoch = epoch, skip_plot = False, batch_size=65536, verbose = True)
            
            # Define file directory
            file_dir = "/home/jsoh/IS3107"
            torch.save(model.state_dict(), '/home/jsoh/IS3107/resale-a2-'+str(combi["a2"])+'-c-'+str(combi["c"])+'.pt')



    def last_task(**kwargs):
        # Some code to execute
        print("end")
        

    encode_task = PythonOperator(task_id='encode', python_callable=encode)
    hyper_param_construction_task = PythonOperator(task_id='hyper_param_construction', python_callable=hyper_param_construction)
    hyper_param_resale_task = PythonOperator(task_id='hyper_param_resale', python_callable=hyper_param_resale)

    do_last_task = PythonOperator(
        task_id='do_last_task',
        python_callable=last_task,
        trigger_rule='one_success'
        # trigger_rule="none_failed"
    )

    # get_new_data_task >> [get_raw_data_task, add_lat_long_task, do_last_task]

    encode_task >> hyper_param_construction_task >> hyper_param_resale_task >> do_last_task

    
