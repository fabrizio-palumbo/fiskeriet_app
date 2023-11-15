import streamlit as st
from pyjstat import pyjstat
import requests
import json, os
import pandas as pd
import streamlit as st
import urllib.request, json
from custom_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
import pandas as pd

def import_dataset_ERS(year_start, year_stop,type_fisks):
    
    dfs = []
    for year in tqdm(range(year_start, year_stop+1)):
        df = pd.read_csv(f'/Users/fabrizio/Documents/GitHub/fiskeri-data/raw_data/ERS/elektronisk-rapportering-ers-{year}-fangstmelding-dca.csv', delimiter=';', 
                        decimal=',', low_memory=False)
        dfs.append(df)#[df['Art - FDIR'].isin(type_fisks)])
        
        #dfs.append(df)
    dfs_concat=pd.concat(dfs)
    typefisk=list(set(dfs_concat['Art - FDIR']))
    dfs_concat['Kvotebelastet reg.merke']=dfs_concat["Registreringsmerke"]
    #dfs_concat['Kvotebelastet reg.merke'].fillna(dfs_concat["Registreringsmerke"], inplace=True)

    #dfs_concat=dfs_concat.query("`Art - FDIR` in @type_fisks")
    
    for col in ['Starttidspunkt', 'Stopptidspunkt']:
        dfs_concat[col] = pd.to_datetime(dfs_concat[col], dayfirst=True, format='mixed')
    return dfs_concat,typefisk

def plot_dca_map_catch_interactive_year_old(df_tot, leng_or_komp, color_scale,marker_opacity=0.7, output_html_path="figure.html", variable_size="Rundvekt"):

    if 'year' not in df_tot.columns:
        raise ValueError("Dataframe must contain 'year' column")

    # Group by the necessary columns and sum the catch sizes
    grouped = df_tot[['year', "Art", 'Startposisjon bredde', 'Startposisjon lengde', variable_size]].groupby(
            by=['year', "Art", 'Startposisjon bredde', 'Startposisjon lengde']).sum().reset_index()

    # Rename columns to match expected names for the scatter_mapbox function
    grouped = grouped.rename(columns={
        'Startposisjon bredde': 'LATITUDE',
        'Startposisjon lengde': 'LONGITUDE'
    })

    # Make sure color_discrete_map has an entry for each year
    years = sorted(grouped['year'].unique())
    for year in years:
        print(year)
        #if year not in color_discrete_map:
        #    raise ValueError(f"No color provided in color_discrete_map for year {year}")
    # Create a continuous color scale
    min_year = grouped['year'].min()
    max_year = grouped['year'].max()
    #color_scale = px.colors.make_colorscale([color_discrete_map[min_year], color_discrete_map[max_year]])

    # Create the animated map using Plotly Express
    fig = px.scatter_mapbox(grouped, lat='LATITUDE', lon='LONGITUDE', 
                            color='year',  # Now using 'year' for color
                            size=variable_size,
                            color_continuous_scale=color_scale,  # Use the continuous color scale
                            range_color=[min_year, max_year],  # Set the range of the color scale

                            animation_frame="year",
                            animation_group="Art",
                            size_max=15, zoom=1,
                            hover_name="Art", hover_data=["year", variable_size])
    # Set marker opacity
    #fig.update_traces(marker=dict(opacity=marker_opacity))
    # Update layout with mapbox style and title
    #fig.update_layout(
    #    mapbox_style='open-street-map',#"stamen-terrain",#"white-bg",#
    #    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    #    title=f'Catch over the Years by {grouped.Art.unique()[0]}',
    #)
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ])
    # Add title
    fig.add_annotation(
        text=f"DCA over the Years by {grouped.Art.unique()[0]}",
        x=0.5, y=1, xref="paper", yref="paper",
        showarrow=False, font=dict(size=18, color="black"),
        align="left"
    )

    # Save the figure as an HTML file
    #fig.write_html(output_html_path)

    # Display the map
    fig.show()

year_start = 2011; year_stop = 2022
list_fleet_type=pd.read_excel('/Users/fabrizio/Documents/GitHub/fiskeri-data/raw_data/fartøyinndeling_kompensasjon.xlsx')
typefisks=['Torsk', 'Sei', 'Dypvannsreke']#,'Reke av  Palaemonidaeslekten',  'Hestereke', 'Reke av  Pandalusslekten', 'Reke av Penaeusslekten', 'Reke av Crangonidaeslekten']
cwd = os.getcwd()

db_folder=cwd+"/database/"
#ers_fisk_tot = pd.read_parquet("Ers_fisk_total.parquet")
ers_fisk = pd.read_parquet("db_folder/Ers_fisk.parquet")
lenge_grouppe=['28 m og over',"27,99 m og under"]#,'Annet'
mapping_dict_lenge = {'28 m og over':'28 m og over','21-27,99 m': "27,99 m og under",
'15-20,99 m': "27,99 m og under",'Under 11 m': "27,99 m og under", '11-14,99 m': "27,99 m og under" }
#ers_fisk_tot["Lengdegruppe"]=ers_fisk_tot["Lengdegruppe"].map(mapping_dict_lenge)
ers_fisk["Lengdegruppe"]=ers_fisk["Lengdegruppe"].map(mapping_dict_lenge)


#plot_dca_map_catch_heatmap(temp, leng_or_komp="Lengdegruppe",color_discrete_map=color_map_Lenge, output_html_path="images/"+"Total_norsk_catch_Lengdegruppe_"+fish_type+".html")




st.title('Norwegian Fishing activities')
title_container = st.container()
with title_container:
    # Define the column layout
    col1, col2 = st.columns(2)
      # Add the webpage to the first column
    with col2:
        l = st.selectbox('Select the Lengegruppe',options= [k for k in lenge_grouppe])     
        fish_type = st.selectbox('Select the fish type',options= [k for k in typefisks])     
        temp=ers_fisk.query("Art==@fish_type and Lengdegruppe==@l")
    # Add some other content to the second column
    with col1:
        # Set up the app header
        
        st.write('Welcome to the Norwegian Municipalities Health System Dashboard! This dashboard is designed to help you explore data on the health systems of Norwegian municipalities, with particular focus on Elderlycare.')
        f=plt.figure()
        plot_dca_map_catch_interactive_year_old(temp, leng_or_komp="Lengdegruppe",color_scale='Inferno',marker_opacity=0.3)
        plt.title("histogram of average %increase of all municipalities")
        st.pyplot(f)
        # Add a picture of Norway
