#Note: Please have all 3 files (main.py, resale.py, construction.py) and 1 folder (ML-prediction) in the same directory
#Note: Replace ddir with your directory path
#Disclaimer: We have loaded all our files from csv for your convenience, even though our data will exist in MySQL database instead.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
from geopy.distance import geodesic as GD
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from construction import predict_construction as pred_const
from resale import predict_resale as pred_res
import random
import json

ddir = f"https://raw.githubusercontent.com/foxiegh/IS3107-Project/main/" #Replace with your directory
ml_pred = f'{ddir}/ML-prediction'


@st.cache_data
def load(file):
    df = pd.read_csv(file)
    return df

def groupfunc(x):
    x = x.split()
    x = float(x[0])
    if x < 1:
        return "0-1"
    elif x < 2:
        return "1-2"
    elif x < 3:
        return "2-3"
    else:
        return ">3"
    
@st.cache_data
def bar(data):
    data = data.assign(grouped_by_distance_to_nearest_mrt = data["distance_to_nearest_mrt"].map(lambda x: groupfunc(x)))
    return data

@st.cache_resource
def display_points(x_var, y_var):
    chart = alt.Chart(data).mark_point().encode(
        x=alt.X(x_var, title = f'{x_var}'),
        y=alt.Y(y_var, title = f'{y_var}')
    ).properties(
        title = f'Graph of {y_var.title()} against {x_var.title()}',
        height = 700
    )
    st.altair_chart(chart, use_container_width=True)

@st.cache_resource
def display_bar(x_var, y_var):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(x_var, title = f'{x_var}'),
        y=alt.Y(y_var, title = f'{y_var}')
    )
    st.altair_chart(chart, use_container_width=True)

def data_exploration(data):
    st.header("Data Head")
    st.write(data.head())
    st.header("Data Description")
    st.write(data.describe())
    numeric_col = ["floor_area_sqm", "resale_price", "lease_commence_date", "remaining_lease",\
                    "distance_to_nearest_mrt"] #resale_days_since_2017, "number_sold_in_same_town_month"
    correlation = data[numeric_col].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation,annot=True, ax=ax)
    st.header("Data Correlation")
    st.pyplot(fig)


data = load(f'{ddir}/closest_mrt_df.csv') ## Read from MySQL instead ##
data = bar(data)
mrt_data = load(f'{ddir}/mrtdf.csv')
construction_data = load(f'{ddir}/under_construction_removed_duplicates.csv')

#Start of Pages
st.title("IS3107 Resale Housing Pricing Visualisation and ML Modelling")

page = st.sidebar.radio("Pages", ["Dataset Exploration", "Explore Historical Data", \
                                  "Predict HDB Pricing (Resale)", "Predict HDB Pricing (Under Construction)"])


if page == "Dataset Exploration":
    data_exploration(data)
elif page == "Explore Historical Data":
    choose_x = st.selectbox("Select x-variable", ["town", "floor_area_sqm", "grouped_by_distance_to_nearest_mrt", "flat_model", "flat_type"])
    choose_y = st.selectbox("Select y-variable", ["resale_price"])
    if choose_x == "grouped_by_distance_to_nearest_mrt":
        for_vis = [[], [], [], []]
        for index, row in data.iterrows():
            cat = row[-1]
            if cat == "0-1":
                for_vis[0].append(row[5])
            elif cat == "1-2":
                for_vis[1].append(row[5])
            elif cat == "2-3":
                for_vis[2].append(row[5])
            else:
                for_vis[3].append(row[5])
        fig2, ax2 = plt.subplots()
        ax2.boxplot(for_vis)
        ax2.set_xticklabels(["0-1", "1-2", "2-3", ">3"])
        ax2.set_xlabel("Distance to Nearest MRT (km)")
        ax2.set_ylabel("Resale Price ($)")
        ax2.set_title("Boxplot for Resale Price vs Distance to Nearest MRT")
        st.pyplot(fig2)

    else:
        display_points(choose_x, choose_y)
elif page == "Predict HDB Pricing (Resale)":
    st.header("For Prediction")
    towns = ["ANG MO KIO","BEDOK","BISHAN","BUKIT BATOK","BUKIT MERAH","BUKIT PANJANG","BUKIT TIMAH",\
             "CENTRAL AREA","CHOA CHU KANG","CLEMENTI","GEYLANG","HOUGANG","JURONG EAST","JURONG WEST",\
                "KALLANG/WHAMPOA","MARINE PARADE","PASIR RIS","PUNGGOL","QUEENSTOWN","SEMBAWANG","SENGKANG",\
                    "SERANGOON","TAMPINES","TOA PAYOH","WOODLANDS", "YISHUN"]
    town = st.selectbox("Select Town", towns)

    models = ["Improved", "New Generation","DBSS","Standard","Apartment",\
                "Simplified","Model A","Premium Apartment","Adjoined flat",\
                    "Terrace","Model A-Maisonette","Maisonette","Type S1","Type S2","Model A2",\
                       "Improved-Maisonette","Premium Maisonette","Multi Generation",\
                            "Premium Apartment Loft"]
    
    flat_type = st.selectbox("Select Flat Type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"])
    if (flat_type == "2 ROOM"):
        models.append("2-room")
    types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
    flat_type = types.index(flat_type) + 1
    flat_model = st.selectbox("Select Flat Model", models)
    floor_area_sqm = st.number_input("Select Floor Area", min_value = 0.0, max_value=300.0 , value = 52.23)
    remaining_lease = st.number_input("Remaining Lease", min_value = 0, max_value = 99, value = 99) * 12
    storeys = ['01 TO 03', '04 TO 06', '07 TO 09','10 TO 12', '13 TO 15', 
       '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33',
       '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
    if flat_model == "Terrace":
        storeys = ['01 TO 03']
    storey_range = st.selectbox("Select Storey Range", storeys)
    storey = storeys.index(storey_range) + 1
    number_sold_around_same_mrt = st.number_input("Select Number of HDBs sold around same MRT in the same month", min_value = 0, max_value = 99, value = 0)
    address = st.text_input("Search for Address (Please make sure the town is correct, otherwise results will be inaccurate)", value = f'{town} MRT')
    response_data = requests.get("https://developers.onemap.sg/commonapi/search?searchVal=" + address + "&returnGeom=Y&getAddrDetails=N&pageNum=1")
    nearest_mrt = requests.get("https://developers.onemap.sg/commonapi/search?searchVal=" + f'{town} MRT' + "&returnGeom=Y&getAddrDetails=N&pageNum=1")
    try:
        data2 = json.loads(response_data.text)
        latitude = data2['results'][0]['LATITUDE']
        # Retrieve the longitude value
        longitude = data2['results'][0]['LONGITUDE']
        #mrt_latitude = data1['results'][0]['LATITUDE']
        #mrt_longitude = data1['results'][0]['LONGITUDE']
        #st.write(f'Latitude: {latitude}')
        #st.write(f'Longtitude: {longitude}')
        selected_place = (latitude, longitude)
        closest_mrt = ""
        least_dist = 900.0
        for index, row in mrt_data.iterrows():
            mrt_latitude = row[2]
            mrt_longitude = row[3]
            mrt = (mrt_latitude, mrt_longitude)
            distance = GD(selected_place, mrt).km
            if distance < least_dist:
                least_dist = distance
                closest_mrt = row[1]
        st.write(f'Closest MRT: {closest_mrt}')
        st.write(f'Distance to nearest MRT: {round(least_dist, 2)}km')
        datenow = date.today() - date(2017,1,1)
    
        if address:
            resale_price = pred_res(town = town, flat_type = flat_type, flat_model = flat_model, floor_area_sqm = floor_area_sqm, remaining_lease = remaining_lease,\
                                    storey_range = storey, latitude = latitude, longitude = longitude, nearest_mrt = closest_mrt, distance_to_nearest_mrt = least_dist, \
                                        resale_days_since_2017=datenow.days, number_sold_around_same_mrt_for_month = number_sold_around_same_mrt, file_dir=ml_pred)
            st.header("Prediction")
            st.write(f'Predicted Resale Price: __${resale_price.values[0][0]:,.2f}__')
        st.header("Further Analysis for Trends")
        choose_x_const = st.selectbox("Select a x-axis value", ["Floor Area", "Storey Range", "Distance to Nearest MRT"])
        if choose_x_const == "Storey Range":
            storeys = [x+1 for x in range(17)]
            resale_prices = pred_res(town = [town]*17, flat_type = [flat_type]*17, flat_model = [flat_model]*17, floor_area_sqm = [floor_area_sqm]*17, remaining_lease = [remaining_lease]*17,\
                                    storey_range = storeys, latitude = [latitude]*17, longitude = [longitude]*17, nearest_mrt = [closest_mrt]*17, distance_to_nearest_mrt = [least_dist]*17, \
                                        resale_days_since_2017=datenow.days, number_sold_around_same_mrt_for_month = number_sold_around_same_mrt, file_dir=ml_pred, multi_pred=True)
            
            storeys = ['01 TO 03', '04 TO 06', '07 TO 09','10 TO 12', '13 TO 15', 
        '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33',
        '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
            fig3, ax3 = plt.subplots()
            ax3.plot(storeys, resale_prices)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation = 90)
            ax3.set_xlabel("Storey Range")
            ax3.set_ylabel("Resale Price($)")
            st.pyplot(fig3)
        elif choose_x_const == "Floor Area":
            rand_points = [random.randrange(0, 300) for x in range(50)]
            resale_prices = pred_res(town = [town]*50, flat_type = [flat_type]*50, flat_model = [flat_model]*50, floor_area_sqm = rand_points, remaining_lease = [remaining_lease]*50,\
                                    storey_range = [storey]*50, latitude = [latitude]*50, longitude = [longitude]*50, nearest_mrt = [closest_mrt]*50, distance_to_nearest_mrt = [least_dist]*50, \
                                        resale_days_since_2017=datenow.days, number_sold_around_same_mrt_for_month = number_sold_around_same_mrt, file_dir=ml_pred, multi_pred=True)
            st.write("We are merely showing the trend using 50 randomly generated values of Floor Area. Rerun to see a slightly different graph!")
            fig4, ax4 = plt.subplots()
            ax4.scatter(rand_points,resale_prices)
            ax4.set_xlabel("Floor Area (sqm)")
            ax4.set_ylabel("Resale Price($)")
            st.pyplot(fig4)
        elif choose_x_const == "Distance to Nearest MRT":
            rand_points = [random.random() * 3 for x in range(50)]
            resale_prices = pred_res(town = [town]*50, flat_type = [flat_type]*50, flat_model = [flat_model]*50, floor_area_sqm = [floor_area_sqm] * 50, remaining_lease = [remaining_lease]*50,\
                                    storey_range = [storey]*50, latitude = [latitude]*50, longitude = [longitude]*50, nearest_mrt = [closest_mrt]*50, distance_to_nearest_mrt = rand_points, \
                                        resale_days_since_2017=datenow.days, number_sold_around_same_mrt_for_month = number_sold_around_same_mrt, file_dir=ml_pred, multi_pred=True)
            fig5, ax5 = plt.subplots()
            st.write("We are merely showing the trend using 50 randomly generated values of distance to closest MRT. Rerun to see a slightly different graph!")
            ax5.scatter(rand_points, resale_prices)
            ax5.set_xlabel(f'Distance away from {closest_mrt}')
            ax5.set_ylabel("Resale Price($)")
            st.pyplot(fig5)
    except:
        st.write("No Such Address found. Please try again or clear search")
    
    
elif page == "Predict HDB Pricing (Under Construction)":
    st.header("For Prediction")
    estates = ["FENGSHAN GREENVILLE","WEST RIDGES @ BUKIT BATOK","BUKIT GOMBAK VISTA","PUNGGOL BAYVIEW","MARSILING GREENVIEW",\
               "SUN NATURA","TOA PAYOH APEX","WEST TERRA @ BUKIT BATOK","BUANGKOK SQUARE","YUNG HO SPRING I","YUNG HO SPRING II",\
                "ST GEORGE'S TOWERS","SUN BREEZE","NORTHSHORE RESIDENCES I","NORTHSHORE RESIDENCES II","WEST QUARRY @ BUKIT BATOK",\
                    "FERNVALE WOODS","ALKAFF VISTA","ALKAFF LAKEVIEW","TECK WHYE VISTA","ALKAFF COURTVIEW","HOUGANG RIVERCOURT",\
                        "NORTHSHORE STRAITSVIEW","WATERFRONT I @ NORTHSHORE","WATERFRONT II @ NORTHSHORE","ANCHORVALE PLAINS",\
                            "WEST PLAINS @ BUKIT BATOK","ALKAFF OASIS","EASTCREEK @ CANBERRA","BEDOK NORTH WOODS","ANG MO KIO COURT",\
                                "SENJA RIDGES","SENJA HEIGHTS","SENJA VALLEY","TAMPINES GREENVIEW","BUANGKOK WOODS","EASTDELTA @ CANBERRA",\
                                    "VALLEY SPRING @ YISHUN","TAMPINES GREENVERGE","BEDOK NORTH VALE","MATILDA SUNDECK","WATERWAY SUNRISE I",\
                                        "BEDOK SOUTH HORIZON","KALLANG RESIDENCES","BEDOK BEACON","ANCHORVALE FIELDS","BLOSSOM SPRING @ YISHUN",\
                                            "MEADOW SPRING @ YISHUN","TAMPINES GREENRIDGES","BUANGKOK TROPICA","WEST ROCK @ BUKIT BATOK","BUANGKOK PARKVISTA",\
                                                "WEST EDGE @ BUKIT BATOK","MACPHERSON SPRING","EASTLINK II @ CANBERRA","TAMPINES GREENWEAVE","EASTLINK I @ CANBERRA",\
                                                    "CLEMENTI CREST","NORTHSHORE TRIO","WOODLEIGH GLEN","WOODLEIGH VILLAGE","TAMPINES GREENBLOOM","TAMPINES GREENFLORA",\
                                                        "WATERWAY SUNRISE II","NORTHSHORE COVE","CLEMENTI NORTHARC","CLEMENTI PEAKS","WOODLANDS SPRING","PINE VISTA",\
                                                            "DAKOTA BREEZE","WOODLEIGH HILLSIDE","JURONG EAST VISTA","SKYRESIDENCE @ DAWSON","SKYOASIS @ DAWSON","FORFAR HEIGHTS",\
                                                                "CITY VUE @HENDERSON","DAWSON VISTA","SKYPARC @ DAWSON","YISHUN N4 C21A","YISHUN N4 C21B","SEMBAWANG N1 C13","SENGKANG N4 C41"]
    estate = st.selectbox("Select New HDB Estate", estates)

    models = ["Improved", "New Generation","DBSS","Standard","Apartment",\
                "Simplified","Model A","Premium Apartment","Adjoined flat",\
                    "Model A-Maisonette","Maisonette","Type S1","Type S2","Model A2",\
                       "Terrace","Improved-Maisonette","Premium Maisonette","Multi Generation",\
                            "Premium Apartment Loft"]
    
    flat_type = st.selectbox("Select Flat Type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"])
    if (flat_type == "2 ROOM"):
        models.append("2-room")
    types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
    flat_type = types.index(flat_type) + 1
    flat_model = st.selectbox("Select Flat Model", models)
    floor_area_sqm = st.number_input("Select Floor Area", min_value = 0.0, max_value=300.0 , value = 52.23)
    remaining_lease = st.number_input("Remaining Lease", min_value = 99, max_value = 99, value = 99) * 12
    storeys = ['01 TO 03', '04 TO 06', '07 TO 09','10 TO 12', '13 TO 15', 
       '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33',
       '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
    if flat_model == "Terrace":
        storeys = ['01 TO 03']
    storey_range = st.selectbox("Select Storey Range", storeys)
    storeys = storeys.index(storey_range) + 1

    latitude = 0
    longitude = 0
    distance_from_nearest_mrt = 0
    nearest_mrt = ""
    for index, row in construction_data.iterrows():
        if row[1] == estate:
            latitude = row[4]
            longitude = row[5]
            distance_from_nearest_mrt = row[6][0:-3]
            nearest_mrt = row[7]
    
    datenow = date.today() - date(2017,1,1)
    
    const_price = pred_const(flat_type = flat_type, flat_model = flat_model, floor_area_sqm = floor_area_sqm, remaining_lease = remaining_lease,\
                                 storey_range = storeys, latitude = latitude, longitude = longitude, nearest_mrt = nearest_mrt, \
                                    distance_to_nearest_mrt = distance_from_nearest_mrt, resale_days_since_2017=datenow.days, file_dir=ml_pred)
    
    st.header("Prediction")
    st.write(f'Predicted Price of Under Construction Estate: __${const_price.values[0][0]:,.2f}__')



    