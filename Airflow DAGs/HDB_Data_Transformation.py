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

default_args = {
    'owner': 'airflow',
}

with DAG(
    'Projectv4',
    default_args=default_args,
    description='HDB Data tranformation',
    #Can be set to daily to retrieve new data everyday
    schedule_interval="@weekly",
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    def get_raw_data(**kwargs):
        ti = kwargs['ti']
        url = 'https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=10'
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['result']['records'])
        df.to_sql("raw", getEngine(), if_exists = 'replace', index=False)
        print("done")


    def getEngine():
        engine = create_engine("mysql+mysqldb://root:password@192.168.xxx.x:3306/hdb")
        return engine
    

    def check_for_new_data(**kwargs):
        ti = kwargs['ti']
        # Retrieve data from API endpoint
        url = 'https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=10'
        r = requests.get(url)
        data = r.json()
        
        # Load data into a pandas DataFrame
        df_new = pd.DataFrame(data['result']['records'])
        
        # Check if new data is available
        engine = create_engine("mysql+mysqldb://root:password@192.168.xxx.x:3306/hdb")
        inspector = inspect(engine)
        if 'raw' not in inspector.get_table_names():
            print('no raw')
            return 'get_raw_data'
        else:
            ti.xcom_push(key='raw_data_exists', value=True)
            # Retrieve existing data from MySQL database
            with engine.connect() as conn:
                #SQL should get the count which is num of rows
                result = conn.execute("SELECT count(month) FROM raw")
                num_rows = result.fetchone()[0]
        
        if len(df_new) > num_rows:
            # Retrieve existing data from MySQL database
            query = "SELECT * FROM raw"
            df_old = pd.read_sql(query, engine)
            
            # Compare the two DataFrames
            df_diff = pd.concat([df_new, df_old]).drop_duplicates(keep=False)
            
            # Write new data to MySQL database
            if not df_diff.empty:
                df_diff.to_sql('raw', engine, if_exists='append', index=False)
                
                # Log new data
                print(f'Added {len(df_diff)} new records.')
                return 'add_lat_long'
        else:
            print('No new data.')
            return 'do_last_task'


    def add_lat_long(**kwargs):
        df = pd.read_sql_query('''SELECT * FROM raw''', con=getEngine())
        df['address'] = df['block'] + " " + df['street_name']

        lat_list = []
        long_list = []

        #Adding latitude and longitude
        for add in df['address']:
            response_data = requests.get("https://developers.onemap.sg/commonapi/search?searchVal=" + add + "&returnGeom=Y&getAddrDetails=N&pageNum=1")

            # Parse response data into a dictionary
            data2 = json.loads(response_data.text)

            # Retrieve the latitude value
            latitude = data2['results'][0]['LATITUDE']
            # Retrieve the longitude value
            longitude = data2['results'][0]['LONGITUDE']

            lat_list.append(latitude)
            long_list.append(longitude)
        
        df['latitude'] = lat_list
        df['longitude'] = long_list

        df.to_sql("latlong", getEngine(), if_exists = 'replace', index=False)

    def last_task(**kwargs):
        # Some code to execute
        print("end")



    
    def get_mrt_addr(**kwargs):

        if (not getEngine().has_table(table_name="mrt", schema="hdb")):
            # the new file does not include ten mile junction station
            staionCodeNew = "https://docs.google.com/spreadsheets/d/1mDJYBkW-NRj-5-PQIw9CqPnAJZMnHld-/export"
            mrtdf = pd.read_excel(staionCodeNew, engine='openpyxl')
            mrtdf = mrtdf.drop(columns = ['mrt_station_english', 'mrt_station_chinese', 'mrt_line_english', 'mrt_line_chinese'])

            #Create Latitude and longitude columns for the mrt using OneMap API

            mrtURL = "https://developers.onemap.sg/commonapi/search?returnGeom=Y&getAddrDetails=N&searchVal="

            mrtLat = []
            mrtLong = []

            mrtList = mrtdf['stn_code'].values.tolist()

            for code in mrtList:
                #API Call
                response_data = requests.get(mrtURL + code)
                data3 = json.loads(response_data.text)

                #print(code)
                # Retrieve the latitude value
                latitude = data3['results'][0]['LATITUDE']
                # Retrieve the longitude value
                longitude = data3['results'][0]['LONGITUDE']

                mrtLat.append(latitude)
                mrtLong.append(longitude)

            print(len(mrtLat))

            mrtdf['latitude'] = mrtLat
            mrtdf['longitude'] = mrtLong
            mrtdf.to_sql("mrt", getEngine(), if_exists = 'replace', index=False)
        
        else:
            print("db alr exists")

    def add_nearest_mrt(**kwargs):
        nearestMrtList = []
        distToNearestMrt = []
                
        df = pd.read_sql_query('''SELECT * FROM latlong''', con=getEngine())
        mrtdf = pd.read_sql_query('''SELECT * FROM mrt''', con=getEngine())

        for index, rows in df.iterrows():
            #print(df['latitude'][index])
            selectedHDB =(df['latitude'][index],df['longitude'][index])

            #choose the first mrt as a base comparison
            nearestMrt = mrtdf['stn_code'][0]
            shortestDist = GD(selectedHDB, (mrtdf['latitude'][0],mrtdf['longitude'][0]) )

            for index2, rows2 in mrtdf.iterrows():
                selectedMrt = (mrtdf['latitude'][index2],mrtdf['longitude'][index2])
                distToSelectedMrt = GD(selectedHDB, selectedMrt)
                if shortestDist > distToSelectedMrt:
                    nearestMrt = mrtdf['stn_code'][index2]
                    shortestDist = distToSelectedMrt
                
            nearestMrtList.append(nearestMrt)
            distToNearestMrt.append(shortestDist)

        df['nearest_mrt'] = nearestMrtList
        df['distance_to_nearest_mrt'] = distToNearestMrt

        df.to_sql("closest_mrt", getEngine(), if_exists = 'replace', index=False)
        
        


    get_raw_data_task = PythonOperator(task_id='get_raw_data', python_callable=get_raw_data)

    add_lat_long_task = PythonOperator(task_id='add_lat_long', python_callable=add_lat_long, trigger_rule='one_success')

    get_mrt_addr_task = PythonOperator(task_id='get_mrt_addr', python_callable=get_mrt_addr)

    add_nearest_mrt_task = PythonOperator(task_id='add_nearest_mrt', python_callable=add_nearest_mrt)


    get_new_data_task = BranchPythonOperator(task_id='check_for_new_data', python_callable=check_for_new_data, depends_on_past=False,
    wait_for_downstream=False, trigger_rule='one_success')

    do_last_task = PythonOperator(
        task_id='do_last_task',
        python_callable=last_task,
        trigger_rule='one_success'
    )

    get_new_data_task >> get_raw_data_task
    get_new_data_task >> add_lat_long_task
    get_new_data_task >> do_last_task
    # get_new_data_task >> [get_raw_data_task, add_lat_long_task, do_last_task]

    get_raw_data_task >> add_lat_long_task >> get_mrt_addr_task >> add_nearest_mrt_task >> do_last_task

    
