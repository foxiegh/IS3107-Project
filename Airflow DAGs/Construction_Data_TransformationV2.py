#use !pip install xxxx if theres any package error
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from geopy.distance import geodesic as GD
from sqlalchemy import create_engine
from sqlalchemy import inspect
import requests
import json
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

default_args = {
    'owner': 'airflow',
}

with DAG(
    'Projectv5',
    default_args=default_args,
    description='HDB Data tranformation',
    schedule_interval="@weekly",
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    
    def getEngine():
        engine = create_engine("mysql+mysqldb://wsl_root:password@172.20.xxx.x:3306/hdb")

        return engine

    # Function to parse coordinates string
    def parse_coordinates(coordinates):
        # Split coordinates string into a list of coordinates
        coordinates_list = coordinates.split()
        # Extract latitude and longitude from each coordinate
        latitudes = [float(coord.split(',')[1]) for coord in coordinates_list]
        longitudes = [float(coord.split(',')[0]) for coord in coordinates_list]
        # Return a tuple of latitude and longitude lists
        return latitudes, longitudes

    # Function to find nearest MRT
    def nearest_MRT_fn(latitude, longitude):
        mrtdf = pd.read_sql_query('''SELECT * FROM mrt''', con=getEngine())
        selectedHDB =(latitude, longitude)

    #choose the first mrt as a base comparison
        nearestMrt = mrtdf['stn_code'][0]
        shortestDist = GD(selectedHDB, (mrtdf['latitude'][0],mrtdf['longitude'][0]) )

        for index2, rows2 in mrtdf.iterrows():
            selectedMrt = (mrtdf['latitude'][index2],mrtdf['longitude'][index2])
            distToSelectedMrt = GD(selectedHDB, selectedMrt)
            if shortestDist > distToSelectedMrt:
                nearestMrt = mrtdf['stn_code'][index2]
                shortestDist = distToSelectedMrt           

        return shortestDist, nearestMrt
    
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
            print("Table already exists!")


    def get_construction_data(**kwargs):
        # Parse the KML file
        tree = ET.parse('/home/jsoh/IS3107/hdb-public-housing-building-under-construction-kml.kml')

        # tree = ET.parse('construction.kml')
        root = tree.getroot()

        # Define the KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Extract placemark data from KML file
        placemarks = root.findall('.//kml:Placemark', ns)

        # Extract data from each placemark
        data = []
        for placemark in placemarks:
            name = placemark.find('.//kml:SimpleData[@name="NAME"]', ns).text
            description = placemark.find('.//kml:SimpleData[@name="DESCRIPTION"]', ns).text
            coordinates = placemark.find('.//kml:coordinates', ns).text.strip()
            latitudes, longitudes = parse_coordinates(coordinates) #this is lat and long for every hdb in this placemark which is a hdb cluster

            #To find distinct hdb names
            distance_from_mrt, nearest_Mrt = nearest_MRT_fn(latitudes[0], longitudes[0])
            data.append([name, description, coordinates, latitudes[0], longitudes[0], distance_from_mrt, nearest_Mrt])

        # Convert data to pandas DataFrame
        df2 = pd.DataFrame(data, columns=['Name', 'Description', 'Coordinates', 'Latitude', 'Longitude', 'Distance_from_MRT', 'Nearest_MRT'])

        # Preview the DataFrame
        print(df2.drop(["Coordinates"], axis=1).head())
        df2.to_sql("construction", getEngine(), if_exists = 'replace', index=False)

    get_construction_data_task = PythonOperator(task_id='get_construction_data', python_callable=get_construction_data)

    get_mrt_addr_task = PythonOperator(task_id='get_mrt_addr', python_callable=get_mrt_addr)

    get_mrt_addr_task >> get_construction_data_task
