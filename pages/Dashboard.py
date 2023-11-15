# Importing the required libraries

import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from matplotlib import cm, colors
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import pearsonr,spearmanr
cwd = os.getcwd()
list_variables=dict()
list_variables['Hele helse- og omsorgstjenesten']=st.session_state['variables_hele']
list_variables['Hjemmetjeneste']=st.session_state['variables_hjemme']
list_variables['Sykehjem']=st.session_state['variables_sykehjem']
data_kostra=st.session_state['data_kostra']

keys_with_negative_impact = ["Brukerer_67+_syk"]


def plot_graph_kommune(dataframe_kom,dataframe_mean_kostra,kom_name,year,y_label):
    dataframe_mean_kostra=dataframe_mean_kostra.replace(np.inf, np.nan)
    
    df_plot = pd.DataFrame({kom_name:dataframe_kom,'kostra_mean':dataframe_mean_kostra.mean(axis=0)
    ,'Year':  list(year)
    })
    band_plot = dataframe_mean_kostra.melt( value_vars=year, var_name="Year", value_name=y_label, col_level=None, ignore_index=True)    
    
    df_plot_kom_meankostra = df_plot.melt('Year', var_name='name', value_name=y_label)
    line = alt.Chart(df_plot_kom_meankostra).mark_line(strokeWidth=10).encode(
    alt.X('Year',scale=alt.Scale(zero=False)),
    alt.Y(y_label,scale=alt.Scale(zero=False)),
    # ,color=alt.Color("name:N")
    color= alt.Color('name',
                   scale=alt.Scale(
            domain=['kostra_mean', kom_name],
            range=['yellow', 'magenta'])))
    band = alt.Chart(band_plot ).mark_errorband(extent='ci', color='yellow'
    ).encode(
    x='Year',
    y= y_label,
    #color= "steelblue"
    )
    chart=alt.layer(band ,line).properties(
        height=450,# width= 450
        autosize = "fit",
        
      #   title=stock_title
    ).configure_title(
        fontSize=16
    ).configure_view(
    strokeWidth=0
).configure_axis(
        grid=False,
        titleFontSize=24,
        labelFontSize=14
    ).configure_legend(
    orient='top-left',
    labelFontSize=20,
)
    return chart#line, band 


years_list=["2020","2021","2022"]
from PIL import Image
folder_icons=cwd+"/icons/"
icons = []
icons.extend([Image.open(folder_icons+'psychotherapy-fill.png')])
icons.extend([Image.open(folder_icons+'bxs-time-five.png')])
icons.extend([Image.open(folder_icons+'nutrition.png')])

face=[]
face.extend([Image.open(folder_icons+'sad.png')])
face.extend([Image.open(folder_icons+'face.png')])
face.extend([Image.open(folder_icons+'happy.png')])

face_inverted=[]
face_inverted.extend([Image.open(folder_icons+'happy.png')])
face_inverted.extend([Image.open(folder_icons+'face.png')])
face_inverted.extend([Image.open(folder_icons+'sad.png')])

arrow_normal=[]
arrow_normal.extend([Image.open(folder_icons+'down.png')])
arrow_normal.extend([Image.open(folder_icons+'flat.png')])
arrow_normal.extend([Image.open(folder_icons+'up.png')])

arrow_inverted=[]
arrow_inverted.extend([Image.open(folder_icons+'down_good.png')])
arrow_inverted.extend([Image.open(folder_icons+'flat.png')])
arrow_inverted.extend([Image.open(folder_icons+'up_bad.png')])

def main():
    scope=st.selectbox('Select the Scope of your investigation:',options= list_variables.keys())     
    keys_selection=list(list_variables[scope].keys())
    print(list_variables[scope][keys_selection[0]])
    list_kom_names=list(set(list_variables[scope][keys_selection[0]].region))
    #with st.sidebar:   
    # ------------------------------------------------------------------------
    # Koumne dropdown list 
    komune_name = st.selectbox('Select the komune name',options= list_kom_names)     
    keys_with_negative_impact = ["Sykefravær","Brukerer_67+_syk","Ventetid på sykehjemsplass"]
    
    
    komune_code_query = list_variables[scope][keys_selection[0]].query("region == @komune_name")["kode"]
    kommune_kode=komune_code_query.iloc[0]
    #print(kommune_kode)
    kom_gruppe = data_kostra.loc[int(kommune_kode)]['kostragr']
    
    list_kom_kostra = list(data_kostra.query('kostragr == @kom_gruppe').index)
    
    list_komune_kostra = [str(w) for w in list_kom_kostra]        
    

    
    c=[]
    for i,values in  enumerate(keys_selection):        
        text1=values
        text2="Trend in the last 3 years"
        text3="Compared to National Statistic"

        if(values in keys_with_negative_impact):
            arrow_temp=arrow_inverted
            up_or_down=face_inverted
        else:
            arrow_temp=arrow_normal
            up_or_down=face
        data = list_variables[scope][values]
        st.write(data)
        try:
            data=pd.pivot_table(data, values="value", columns="år",index="kode")#["år", "statistikkvariabel"]
        except:
            data=pd.pivot_table(data, values="Value", columns="år",index="OrgNumber")
        data=data[years_list]
        
        try:
            dataset = data.loc[kommune_kode]
            #mean_education=data_education.loc[list_komune_kostra].median(axis = 0)
            National_75th=data.quantile(q=0.75,axis = 0)
            National_25th=data.quantile(q=0.25,axis = 0)
            dataset=dataset.loc[years_list]
            c.extend([st.container()])
            with c[-1]:
                cols=st.columns(3)    
                with cols[0]:
                    st.title(text1)
                    #st.image(icons[0],use_column_width='always')
                    #print(data.index)
                    kostra_mean=data.loc[[ str(komune_index) for komune_index in list_kom_kostra if str(komune_index) in data.index]]#.median(axis = 0)
                    
                    line_plot=plot_graph_kommune(dataset,kostra_mean,komune_name,years_list, values)  
                    
                    st.altair_chart(line_plot, use_container_width=True)
           
                with cols[1]:
                    st.title(text2)            
                    #diff=dataset.diff(periods=1 )
                    diff=dataset.pct_change(periods=1 ).sum()
                    if(diff>0.02):
                        st.image(arrow_temp[2] ,use_column_width='always')
                    else:
                        if(diff<-0.02):
                            st.image(arrow_temp[0],use_column_width='always')
                        else:
                            st.image(arrow_temp[1],use_column_width='always')
                with cols[2]:
                    st.title(text3)            
                    if(dataset[-1]>National_25th[-1]):
                        if(dataset[-1]>National_75th[-1]):
                            st.image(up_or_down[2],use_column_width='always')
                        else:
                            st.image(up_or_down[1],use_column_width='always')
                    else:
                        st.image(up_or_down[1],use_column_width='always')

        except Exception as error:
                st.write("We miss some index value for this kom", komune_name, "Place Name :"+ komune_name,"->" + values)
                st.write(error.args)
                st.write(dataset)
    
    
    


    return


    
main()

