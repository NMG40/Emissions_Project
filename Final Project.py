#!/usr/bin/env python
# coding: utf-8

# #  An Analysis of the Impact of Household Size on Carbon Footprints
# ## Project One

# ## 1.1 Introduction
# 
# Given the current climate emergency that has been afflicting the planet there has been a growing interest in using data to identify the most effective solutions to mitigate the effects of the changing temperatures. This current paper will be a part of a series of papers exploring the issue, specifically in the realm of city planning and how good urbanization may lead to lower carbon emissions without necessarily requiring the degrowth policies many politicians fear espousing. 
# 
# In this paper, I will analyze data recovered from the Global Carbon Project, to determine the general distribution of carbon emissions across countries, across time periods and the correlation each of them have with total emissions. Then in the second part of the project, I recovered data from the United Nations Department of Economic and Social Affairs that recorded household size data for a number of different countries at different times.

# ## Set Up
# 
# In this projet, we will examine first: the relationship between the six carbon emission sources (coal, oil, gas, cement, flaring, other) and the total, then between the six and Per Capita emissions. This will be the basis for further research pertaining to the impact of household size on carbon footprints
# 
# I am choosing to analyze all the variables in the dataset, which in this case are the six explanatory variables (Coal, Oil, Gas, Cement, Flaring and Other), as well as the year which is an independent variable, in order to examine the two dependent variables (Total and Per Capita emissions).
# 
# I have decided to choose the six emission sources since through them we can get a better understanding of the composition of the emissions, which can be used to provide more useful results by the end of the study. If cement is main source of emissions in countries with a high emissions per capita, wealthier governments should perhaps switch their priorities from focusing on energy sources to switching the material used in construction.
# 
# Inutitively all the emission sources should have a positive effect on total emissions, as total emissions is equal to the sum of the other emissions, however we will explore whether an increase in the emissions of one source actually decreases the emissions of other sources to the extent where the variable has a null or negative effect on total transmissions. 
# 
# After processing the data, one can conclude that the year, despite not being explanatory, tends to have a positive correlation with total and per capita emissions due to the economic development that most countries have gone through in the 271 year period the data covers.
# 
# Once a breakdown of the carbon emissions data is created we can differentiate between the different carbon emission sources (how each of them affects the total) and can move onto how household size changes the emissions and emission sources.

# # 1.2 Data Cleaning

# In[111]:


import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pandas.plotting import scatter_matrix
from scipy.stats import linregress
import geopandas as gpd
from shapely.geometry import Point
get_ipython().run_line_magic('matplotlib', 'inline')
import qeds
import warnings
import requests
from bs4 import BeautifulSoup
from sklearn import linear_model
from sklearn import tree
get_ipython().system('pip install country_converter --upgrade')
import country_converter as coco
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from IPython.core import ultratb
import sys
from IPython.core.error import UsageError
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.iolib.summary2 import summary_col

get_ipython().system('pip install linearmodels')

# Set up the error handler
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

from sklearn import (
    linear_model, metrics, pipeline, model_selection
)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
get_ipython().system('pip install pycebox')
from pycebox.ice import ice, ice_plot


# ## Importing the Data

# In[2]:


emissions_data=pd.read_csv('Emissions.csv')


# In[3]:


emissions_data


# ## Data Cleaning
# #### Here we are switching NaN values to zero to make them easier to manipulate and then we're eliminating all rows where the total emissions is 0, to ensure the data is not skewed leftwards.

# In[4]:


emissions_data.fillna(0, inplace=True)
emissions_data_clean = emissions_data[emissions_data['Total'] != 0]
emissions_data_clean

emissions_data_clean.loc[emissions_data_clean['Country']=='USA']


# ## Data Cleaning for Part 2

# In[5]:


household_data=pd.read_csv('Worldwide household size.csv', header = 1)
household_data=household_data.drop(household_data[household_data["Average household size (number of members)"] == '..'].index)
household_data=household_data.drop(household_data.columns[43:], axis=1)
household_data=household_data.drop(household_data.columns[1:3], axis=1)


# In[6]:


household_data


# #### What follows below is done to format the date such that it will be according to the year and will be easier to merge with the carbon emissions dataset

# In[7]:


date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
household_data['Reference date (dd/mm/yyyy)'] = pd.to_datetime(household_data['Reference date (dd/mm/yyyy)'], 
                                                               errors='coerce', infer_datetime_format=True)


# In[8]:


household_data['Reference date (dd/mm/yyyy)'] = pd.to_datetime(household_data['Reference date (dd/mm/yyyy)'], 
                                                               format='%Y-%m-%d')
household_data['Year'] = household_data['Reference date (dd/mm/yyyy)'].dt.year
household_data


# In[9]:


column_names = list(household_data.columns)
column_names.insert(1, column_names.pop(41))
household_data = household_data.reindex(columns=column_names)
household_data.rename(columns={'Country or area': 'Country'}, inplace=True)
household_data = household_data.set_index("Country")
household_data.rename(index={'United States of America': 'USA'}, inplace=True)
household_data
household_data = household_data.drop(household_data.columns[3:], axis=1)
household_data


# ### Merging Dataframes

# In[10]:


merged_df = pd.merge(emissions_data_clean, household_data, on=['Country', 'Year'])
merged_df = merged_df.drop(merged_df.columns[13:], axis=1)
merged_df = merged_df.set_index("Country")
merged_df


# ## 1.3 Finding the Summary Statistics

# In[11]:


summary_stats = emissions_data_clean[["Total", "Coal", "Oil", "Gas", "Cement", "Flaring", "Other", 
                                      "Per Capita"]].describe(percentiles=[0.2, 0.4, 0.6, 0.8]) 
summary_stats


# The chart above indicates the prevalence of different co2 emissions with coal being the highest emitter among all sources, followed by oil, gas, cement, other and flaring. This we can determine by looking at the *mean* column; given all sources are represented the same amount of times, the mean can be used to rank the different source emissions by contribution to total emissions, across time.
# 
# The standard deviation for every variable is also several orders of magnitude larger than the means per variable, this means that there has been significant variation in the emission numbers and it would therefore be difficult to fit the emissions accurately to a linear regression.
# 
# In addition, if we compare the means of the different variables to their values at the different percentiles, we can see that in almost every case (save Per Capita emissions) the 80th percentile is still lower than the mean. This means that relatively few data points are responsible for affecting the total amount of emissions of every variable (including total emissions). In other words, this means there's a big inequality in the emissions from every source and if we look at the graph below we'll be able to see that it's due to an exponential increase in emissions (from every source) in the past century$^{2}$
# .
# 
# 

# ## 1.4 Charts
# ### $^{2}$ Graphs displaying exponential increase in emissions from different sources, in total and per capita
# 

# In[12]:


emissions_data_Global = emissions_data_clean[emissions_data_clean["Country"]== "Global"]
emissions_data_Global = emissions_data_Global[emissions_data_Global['Year'] >= 1850]

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
axs = axs.flatten()

for i in range(len(emissions_data.columns)):
    if i > 2:
        axs[i-3].plot(emissions_data_Global['Year'], emissions_data_Global[emissions_data.columns[i]])
        axs[i-3].set_xlabel('Year')
        axs[i-3].set_ylabel(emissions_data.columns[i] + " emissions")
        axs[i-3].set_title(emissions_data.columns[i] + " emissions over time")
fig.tight_layout()
plt.show()


# The graphs above are used to display the almost exponential increase in carbon emissions from all sources beginnning in the 20th century. It is also interesting to note the recent stablization of some sources recently, namely from coal and from coal. The per capita chart has also been added in order to illustrate how despite generally total emissions have increased over time, per capita emissions have stabilized and can serve to show how much of the growth in total emissions is not a function of higher more emission inducing consumption but rather sheer population growth.
# 
# There are two notable changes in emissions data: a sudden increase in flaring emissions around 1950, and a rise in emissions from renewable energy sources around 1985. The increase in flaring emissions can be attributed to the fact that flaring became popularized in the mid-20th century, and until 1950 no country had begun to measure emissions from this source. On the other hand, the rise in emissions from other energy sources can be attributed to the growing awareness of carbon emissions from fossil fuels, which led to the increasing popularity of renewable energy sources in the late 20th century. This rise in emissions could come from both the production of the requisite technology, such as photovoltaic cells or wind turbines, and from the production of energy itself, as is the case with biomass.

# ## Further Charts

# The chart below helps to identify the nations that are most responsible for emissions. To create the graph, we first grouped together the G7 nations since they represent the most developed democracies, who are also known for their heavy per capita energy use. We also included the often neglected emissions from international transport, which were surprisingly large and included in the dataset. Finally, we included China as a reference point to highlight their significant impact on total emissions, despite emitting relatively little until the 1950s.

# In[13]:


emissions_sorted = emissions_data_clean.groupby("Country").sum().sort_values("Total", ascending=False)


# In[14]:


emissions_g7 = emissions_sorted.loc[['USA', 'United Kingdom', 'Japan', 'Italy', 'France', 'Germany', 'Canada']]
emissions_g7.head()


# In[15]:


International_transport = emissions_sorted.loc[['International Transport']]
International_transport.head()


# In[16]:


China_emissions = emissions_sorted.loc[['China']]
China_emissions.head()


# In[17]:


Rest_emissions = emissions_sorted.loc[['Global']]
transport_share = International_transport['Total'].sum()/ Rest_emissions['Total']
print(transport_share)


# In[18]:


share_of_g7 = emissions_g7.iloc[1:10]['Total'].sum() / Rest_emissions['Total']
transport_share = International_transport['Total'].sum() / Rest_emissions['Total']
china_share = China_emissions['Total'].sum() / Rest_emissions['Total']

labels = ['G7 Emissions Share', 'International Transport', "China Share", 'Rest of World']
sizes = [share_of_g7, transport_share, china_share, 1 - share_of_g7 - transport_share - china_share]
colors = ['#90EE90', '#FFFF00', '#FF5733', '#ADD8E6']

fig1, ax1 = plt.subplots()
ax1.pie(np.array(sizes).flatten(), colors=colors, labels=labels, startangle=90, frame=True, autopct='%1.1f%%')
fig = plt.gcf()
ax1.axis('equal')
ax1.axis('off')
plt.tight_layout()
plt.title("Shares of Total Emissions")
plt.show()


# The chart below, helps us to visualize the changing sources of total emissions. For about 200 years, we can see that coal was the main source of emissions and by a large margin, but from around 1970 to 2000 there was a boom in oil emissions, most likely related to political events and the growing awareness of coal's consequences. However, once the 2000's hit we can again see a jump in coal emissions as a percentage of the total emissions. What nations were behind this increase in coal emissions?

# In[19]:


emissions_filtered = emissions_data_Global[['Total','Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']]

emissions_filtered = emissions_filtered.iloc[:, 1:].div(emissions_filtered.Total, axis=0) * 100

emissions_filtered = emissions_filtered.assign(Year=emissions_data_Global['Year'])

emissions_filtered.plot(x='Year', y=['Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other'], kind='line')

plt.title("Composition of Total Emissions by Source")

plt.ylabel("Percentage of Total Emissions")

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()


# In the tables below we are seeking to explain who are the main culprits behind coal's resurgence. We find the countries that had the largest absolute increase in coal emissions and then display them on a line graph to help illustrate the trends since 2000 and also show the differences in total emission amounts. 

# In[20]:


emissions_2000 = emissions_data_clean.loc[(emissions_data_clean['Year'] == 2000), ['Country', 'Coal']]
emissions_2021 = emissions_data_clean.loc[(emissions_data_clean['Year'] == 2021), ['Country', 'Coal']]

emissions_2000_2021 = pd.merge(emissions_2000, emissions_2021, on='Country', suffixes=('_2000', '_2021'))
emissions_2000_2021


# In[21]:


emissions_2000 = emissions_data_clean[emissions_data_clean['Year'] == 2000]
emissions_2021 = emissions_data_clean[emissions_data_clean['Year'] == 2021]
emissions_2000_2021 = pd.merge(emissions_2000, emissions_2021, on='Country', suffixes=('_2000', '_2021'))
emissions_2000_2021 = emissions_2000_2021[['Country', 'Coal_2000', 'Coal_2021']]

emissions_2000_2021['Absolute change 2000-2021'] = emissions_2000_2021['Coal_2021'] - emissions_2000_2021['Coal_2000']
emissions_2000_2021['% change 2000-2021'] = emissions_2000_2021['Absolute change 2000-2021'] / emissions_2000_2021['Coal_2000'] * 100

emissions_diff = emissions_2000_2021[['Country', 'Absolute change 2000-2021', '% change 2000-2021']]

mask = emissions_diff['Absolute change 2000-2021'] > 0
emissions_Filtered = emissions_diff[mask]
emissions_Filtered = emissions_Filtered.sort_values(by='Absolute change 2000-2021', ascending=False)
top_30 = emissions_Filtered[1:].head(30)
top_30




# In[22]:


emissions_data_Clean = emissions_data_clean[(emissions_data_clean['Year'] >= 2000) & (emissions_data_clean['Year'] <= 2021)]
emissions_data_Clean = emissions_data_Clean.reset_index()

emissions_china =  emissions_data_Clean[emissions_data_Clean["Country"]== "China"]
emissions_china_coal = emissions_china[["Coal", "Year"]]

emissions_india =  emissions_data_Clean[emissions_data_Clean["Country"]== "India"]
emissions_india_coal = emissions_india[["Coal", "Year"]]

emissions_indo =  emissions_data_Clean[emissions_data_Clean["Country"]== "Indonesia"]
emissions_indo_coal = emissions_indo[["Coal", "Year"]]

emissions_viet =  emissions_data_Clean[emissions_data_Clean["Country"]== "Viet Nam"]
emissions_viet_coal = emissions_viet[["Coal", "Year"]]

emissions_southk =  emissions_data_Clean[emissions_data_Clean["Country"]== "South Korea"] 
emissions_southk_coal = emissions_southk[["Coal", "Year"]] 

plt.plot(emissions_china_coal["Year"], emissions_china_coal["Coal"], label="China")
plt.plot(emissions_india_coal["Year"], emissions_india_coal["Coal"], label="India")
plt.plot(emissions_indo_coal["Year"], emissions_indo_coal["Coal"], label="Indonesia")
plt.plot(emissions_viet_coal["Year"], emissions_viet_coal["Coal"], label="Vietnam")
plt.plot(emissions_southk_coal["Year"], emissions_southk_coal["Coal"], label="South Korea")

plt.title("Total Coal Emissions Growth")
plt.ylabel("Coal Emissions in Metric Tons")
plt.legend()
plt.show()


# China is the biggest culprit behind the coal increases in the past 20 years. In fact, according to the Center for Research on Energy and Clean Air, China is building six times more coal plants than the rest of the world combined ((Myllyvirta et al., China permits two new coal power plants per week in 2022 2023 https://energyandcleanair.org/publication/china-permits-two-new-coal-power-plants-per-week-in-2022/)). Despite this, China is still decreasing the proportion of its energy mix which is accounted for by coal emissions (U.S. Energy Information Administration - EIA - independent statistics and analysis 2022 https://www.eia.gov/international/analysis/country/CHN). 
# 
# One of the challenges facing China and other countries on the list (with the exception of South Korea) is that as they develop and their economies grow, their energy consumption also increases. Coal has traditionally been a cheap and abundant source of energy, and many countries have turned to it to meet their growing energy needs. Despite its negative impact on the environment and human health, coal has been viewed as a reliable and secure source of energy by some countries, leading to its continued use even as renewable energy alternatives become more available.

# # Project Two

# ## 2.1 Message
# The primary objective of this project component is to integrate household size statistics to investigate the correlation between this variable and the previously analyzed carbon emissions. The initial step will involve measuring the relationship between household size and total emissions, followed by examining the correlation between household size in each country and the emissions it produces the most. Finally, countries will be categorized based on specific characteristics to evaluate whether economic conditions and per capita emissions contribute to varying effects of household size on emissions.

# In[23]:


merged_df['Average household size (number of members)'] = pd.to_numeric(
    merged_df['Average household size (number of members)'], errors='coerce')
merged_df['Per Capita'] = pd.to_numeric(merged_df['Per Capita'], errors='coerce')


# In[24]:


merged_df = merged_df.sort_values('Average household size (number of members)')
merged_df = merged_df.rename(columns={'Total':"Total Emissions"})
merged_df.plot.scatter(x = 'Average household size (number of members)', y = 'Per Capita', c = 'Total Emissions', cmap = 'coolwarm')
x = merged_df['Average household size (number of members)'].values
y = merged_df['Per Capita'].values


# add polynomial fit and error band
p = np.polyfit(x, y, 3)
f = np.poly1d(p)
y_pred = np.polyval(p, x)
residuals = y - y_pred
std_error = np.std(residuals)
plt.plot(x, f(x), 'r')
plt.fill_between(x, y_pred-std_error, y_pred+std_error, alpha=0.2, color='orange')

# calculate R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean)**2)
ss_res = np.sum(residuals**2)
r_squared = 1 - ss_res / ss_tot

# set labels and title
plt.xlabel('Average Household Size')
plt.ylabel('Per Capita Emissions in Metric Tons')
plt.title('Per Capita Emissions and Average Household Size')

# add R-squared value to plot
plt.text(0.5, 0.5, f"R-squared: {r_squared:.2f}", transform=plt.gca().transAxes)

plt.savefig('Regression Graph')
plt.show()


# The code above displays the relationship between the average household size, emissions per capita and total emissions. The first thing to note is the uniformity in the colour of the dots. This means that most datapoints have emissions that are relatively low to what the maximum is. This once again reinforces the idea that there is a significant carbon emissions inequality that has been widening throughout the centuries and between countries. In addition, we can see that when household sizes are between 2 and 4 there's an important relationship between the size and per capita carbon emission. This is reinforced by the fact that the per capita emissions stabilizes once household size reaches 4 and is at a point beyond the reach of the standard error. This means that the relationship is most likely significant. Once the household size reaches 4, however, the effect seems to taper off as the line of best fit does not change drastically.
# 
# This effect could exist because of actual house size or endogeneity with per capita GDP. For the first possibility, as described in Katherine Kellsworth-Krebs 2019 paper (Ellsworth-Krebs, Katherine. “Implications of declining household sizes and expectations of home comfort for domestic energy demand.” Nature Energy 5 (2019): 20-25.), the apparent relationship household size has with carbon emissions, may be due more to house size than density. Kellsworth-Krebs found that energy efficiency gains from home maintenance have been offset by the trend of increasing house sizes and energy-intensive lifestyles, particularly among single and couple households in suburban areas with rising housing prices. The main driver of this energy intensive lifestyle has been single and couple housing, as due to the rising trend of subarbanization and rising city housing prices, these populations are buyingg bigger and bigger houses relative to what they were buying before.
# 
# The second possible reason for this relationship could be that many richer countries tend to, for cultural and economic reasons, have citizens that tend to live in smaller groups. In Kellsworth-Krebs' study she looks at the changing attitudes in the Japanese population whereby in the 1950's 60% of women expected to be taken care of by their children, which most of the time was done through cohabitation, while fourty years later this number was 16%. This means that rich countries tend to emit more, but also tend to have smaller households and so the effect of that wealth has on emissions is being captured here by household size serving as a proxy.

# In[25]:


averagehousehold_g7 = merged_df.loc[['USA', 'United Kingdom', 'Japan', 'Italy', 'France', 'Germany']]


# In[26]:


g7_countries = ['USA', 'United Kingdom', 'Japan', 'Italy', 'France', 'Germany']
g7_data = merged_df.loc[g7_countries]

rest_data = merged_df.loc[~merged_df.index.isin(g7_countries)]

g7_mean = g7_data.groupby('Year')['Average household size (number of members)'].mean()
rest_mean = rest_data.groupby('Year')['Average household size (number of members)'].mean()

plt.plot(g7_mean.index, g7_mean, label='G7')
plt.plot(rest_mean.index, rest_mean, label='Rest of the world')

plt.xlabel('Year')
plt.ylabel('Average household size (number of members)')
plt.legend()
plt.title('Average Household Size Throughout the Years')
plt.show()


# We can also explore the differences between household sizes between the G7 and the rest of the world. Here we can see, despite some noisy data, that the G7 has consistently had smaller household sizes than the rest of the world, but as we can see below also consistently have higher emissions per capita.
# 
# We can also see a spike in CO2 emissions from the rest of the world which can partly be explained by the end of the communist revolution in China and the beginning of rapid industrialization in the country, with a 26% increase in emissions from 1949 to 1950. In addition, 1950 was the first year emissions data was available for many developing countries, which also contributed to a small peak at 1950.

# In[27]:


Emissions_data_clean = emissions_data_clean.set_index('Country')
g7_countries = ['USA', 'United Kingdom', 'Japan', 'Italy', 'France', 'Germany']
g7_data = Emissions_data_clean.loc[g7_countries]

rest_data = Emissions_data_clean.loc[~(Emissions_data_clean.index.isin(g7_countries))]
rest_data = rest_data.loc[~(rest_data.index == "International Transport")]
                            

g7_mean = g7_data.groupby('Year')['Per Capita'].mean()
rest_mean = rest_data.groupby('Year')['Per Capita'].mean()

plt.plot(g7_mean.index, g7_mean, label='G7')
plt.plot(rest_mean.index, rest_mean, label='Rest of the world')

plt.xlabel('Year')
plt.ylabel('Average Per Capita Emissions')
plt.legend()
plt.title('Average Per Capita Emissions Throughout the Years')
plt.show()


# ## 2.2 Maps and Interpretation

# Below are two maps. The first showing the distribution of coal emissions and the second displaying average household size in the most recent year the data was collected.
# 
# The share of coal among total emissions is displayed, in order to better understand the relationship between carbon emissions and economic conditions. This was shown on the micro level in South Africa in Balmer's 2017 study of household coal use in urban South Africa (Balmer, M. (2017). Household coal use in an urban township in South Africa. Journal of Energy in Southern Africa, 18, 27-32.).
# 
# The second map is to display the average household size to see how it matches the previous coal emissions map. If we see a trend of countries that are both high in terms of carbon emissions and average household size, then it'll serve to reinforce Balmer's theory on a macro scale. Namely, bigger households burn more coal due to the need to generate more energy than what the state allots. On the other hand, the opposite relationship may be true as well where smaller households consume more, not because household size is directly correlated with emissions but rather because it is correlated with wealth and wealth is correlated with emissions. In further research this relationship will be refined through different empirical methods such as IV tests or a Diff and Diff analysis.

# In[28]:


world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world.set_index("name")

world.head()


# In[29]:


emissions_data["ISO 3166-1 alpha-3"] = emissions_data["ISO 3166-1 alpha-3"].str.title()
emissions_data["ISO 3166-1 alpha-3"] = emissions_data["ISO 3166-1 alpha-3"].str.strip()
world["iso_a3"] = world["iso_a3"].str.title()
world["iso_a3"] = world["iso_a3"].str.strip()
country_emissions = world.merge(emissions_data, left_on="iso_a3", right_on="ISO 3166-1 alpha-3", how="inner")
country_emissions["Coal Share"] = country_emissions["Coal"] / (country_emissions["Total"])


# In[30]:


fig, gax = plt.subplots(figsize = (20,20))
gax.set_xlabel('longitude')
gax.set_ylabel('latitude')

world.plot(ax=gax, edgecolor='black',color='white')

country_emissions.plot(
    ax=gax, edgecolor='black', column='Coal Share', legend=True, cmap='bwr', legend_kwds={"shrink": 0.5},
    vmin=0, vmax=1 
)

gax.annotate('Share of total emissions that come from coal',xy=(0.2, 0.35),fontsize='large', xycoords='figure fraction')

plt.axis('off')

plt.show()


# In[31]:


max_years = household_data.groupby(['Country'])['Year'].max()
max_years = max_years.to_frame().rename(columns= {'Year':'Max_Year'})


# In[32]:


household_data = pd.merge(household_data,max_years, how = 'left', on='Country')


# In[33]:


household_data_max_year = household_data[household_data.Year==household_data.Max_Year]


# In[34]:


world_household = pd.merge(household_data_max_year, world, left_index=True, right_index=True) #ADD MAP


# # Project Three
# ## 3.1 Potential Data to Scrape
# To investigate the impact of household size on carbon emissions and potentially build upon the Kellsworth-Krebs findings, I plan to incorporate floor area per capita data into my analysis. I will run a regression analysis that controls for both square footage per capita and GDP per capita, with the latter being easier to obtain through dataset or web scraping. This will allow me to isolate the effect of household size on carbon emissions.
# 
# To collect floor size per capita, I will scrape information from the websites of various national agencies or real estate websites to gain a better understanding of the factors involved, including historical data. However, despite the abundant sources of data available on the internet, it may be challenging to find consistent and accurate data, particularly in developing countries where house construction is comparatively less regulated, and records pertaining to house size and average occupancy may be scarce.
# 
# In terms of merging, I will merge the floor area per capita data with the existing dataset that already includes household size and emissions. The merging will be based on year and country. I will eliminate any data that isn't represented in any of the three datasets to avoid any errors in the regression analysis. Given the large amount of data available, it is expected that the final dataset will contain more than the required 500 observations for a regression analysis.
# 
# ## 3.2 Potential Challenges
# Several potential challenges may arise with this scraping initiative, including the sheer number of websites that I would need to scrape from, as well as the different metrics and missing data that I may need to collect myself or obtain using satellite data.
# 
# The first issue is self-explanatory. Since there are approximately 200 countries, I would need to gather data from multiple governmental and national real estate websites. Some of these websites also collect their information through self-reported surveys, which may contain imprecise information or may be innacurate due to realtively small sample sizes.  Moreover, much of this data is collected sparsely, making it difficult to obtain.
# 
# Lastly, availability may also pose a challenge. While many countries have data on population density, household size, and housing units, unit size or floor area per capita data may not be readily available. As such, original data collection may need to be done through surveys of residents or construction firms, or rough estimates may need to be made by examining satellite data and population density in a particular region.
# 

# ## 3.3 Scraping the Data
# Instead of embarking on the gargantuan task of collecting enough floor are per capita that would suffice to find a robust relationship i'll scrape GDP per Capita data off of Wikipedia and then add a population density dataset to the dataframe. 
# 
# We'll be using the IMF data from this link ("https://en.wikipedia.org/wiki/List_of_countries_by_past_and_projected_GDP_(nominal)_per_capita") which corresponds to the first four tables on the site. This will gives us each nations gdp per capita for every year from 1980-2019.

# In[35]:


URL = 'https://en.wikipedia.org/wiki/List_of_countries_by_past_and_projected_GDP_(nominal)_per_capita'

# Connect to the URL
Response = requests.get(URL) 

# Parse HTML and save to BeautifulSoup object¶
largest_soup = BeautifulSoup(Response.text, "html.parser") #html.parser telling Beautiful Soup that it's an HTML file


# In[36]:


# Here we will be trying to find the tables in the HTML file by finding the first table classified as "wikitable"
# Using IMF data, so need to take only the first four tables, as opposed to all of the tables on the site
Table_8089 = largest_soup.find_all('table', {'class': 'wikitable'})[0]
header_row = Table_8089.find('tr')
columns = [col.get_text(strip=True) for col in header_row.find_all('th')]


# In[37]:


# Here we're finding all the rows in the first table and for each row we're populating the column with 'td' 
# which is the content in each row
rows = Table_8089.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_8089 = pd.DataFrame(data, columns=columns)


# In[38]:


# Given we have 4 IMF tables we're trying to merge we're doing this process again, this time it's for the years 1990-1999
Table_9099 = largest_soup.find_all('table', {'class': 'wikitable'})[1]
header_row = Table_9099.find('tr')
columns = [col.get_text(strip=True) for col in header_row.find_all('th')]

rows = Table_9099.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_9099 = pd.DataFrame(data, columns=columns)


# In[39]:


# Now for 2000-2009
Table_0009 = largest_soup.find_all('table', {'class': 'wikitable'})[2]
header_row = Table_0009.find('tr')
columns = [col.get_text(strip=True) for col in header_row.find_all('th')]

rows = Table_0009.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_0009 = pd.DataFrame(data, columns=columns)


# In[40]:


# Finally for 2009-2019, note that we're creating different dataframes for the different IMF tables on the site
Table_1019 = largest_soup.find_all('table', {'class': 'wikitable'})[3]
header_row = Table_1019.find('tr')
columns = [col.get_text(strip=True) for col in header_row.find_all('th')]

rows = Table_1019.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df_1019 = pd.DataFrame(data, columns=columns)


# In[41]:


# Now we'll merge all the dataframes

Merged_df = pd.merge(df_8089, df_9099, on= "Country (or dependent territory)")
Merged_df = pd.merge(Merged_df, df_0009, on= "Country (or dependent territory)")
Merged_df = pd.merge(Merged_df, df_1019, on= "Country (or dependent territory)")

Merged_df = Merged_df.rename(columns = {"Country (or dependent territory)": "Country"})
Merged_df = Merged_df.set_index("Country")
Merged_df.rename(index={'United States': 'USA'}, inplace=True)
Merged_df = Merged_df.reset_index(drop=False)


# ## 3.4 Merging 
# First I will merge the data with the original, cleaned emissions dataset, then will merge with the dataset containing household size as well. 
# 
# Given the merged dataframe has the years as the columns instead of column values we'll use pd.melt to create a new "Year" column containing the different years they collected the data for.

# In[42]:


melted_df = pd.melt(Merged_df, id_vars=['Country'], var_name='Year', value_name='GDP Per Capita')
melted_df['Year'] = melted_df['Year'].astype('int64')
melted_df


# In[43]:


emerged_df = pd.merge(emissions_data_clean, melted_df, on=['Country', 'Year'])
emerged_df = emerged_df[emerged_df['GDP Per Capita'] != "—"]

emerged_df


# Note: Although we removed null values, in order to make our tables clearer, we still have 6602 observations. Also note that the GDP per capita is in USD.

# ## 3.5 Visualizations
# 
# In the following we'll create a stacked bar chart to explore whether there have been recent changes in the composition of total emissions based on the GDP per capita of the nation emitting. Then we'll create a hexbin graph to uncover whether there's any potential relationship between GDP per capita and Per capita emissions.
# 
# The former will represent a refining of previous data, using G7 status as a proxy for wealth. It'll also provide a representation of how the composition of total emissions has been changing, according to the wealth of the nation. The bin sizes were chosen to correspond with the relative size of the G7.

# In[44]:


years = [1989, 1999, 2009, 2019]
y1989 = emerged_df[emerged_df['Year']==1989].sort_values(by = ['GDP Per Capita'], ascending = [False])
y1999 = emerged_df[emerged_df['Year']==1999].sort_values(by = ['GDP Per Capita'], ascending = [False])
y2009 = emerged_df[emerged_df['Year']==2009].sort_values(by = ['GDP Per Capita'], ascending = [False])
y2019 = emerged_df[emerged_df['Year']==2019].sort_values(by = ['GDP Per Capita'], ascending = [False])
                         
top_10_89= y1989[:10]
top_10_99= y1999[:10]
top_10_09= y2009[:10]
top_10_19= y2019[:10]

top_1025 = y1989[10:25]
top_1025_99 = y1999[10:25]
top_1025_09 = y2009[10:25]
top_1025_19 = y2019[10:25]

top_2550 = y1989[25:50]
top_2550_99 = y1999[25:50]
top_2550_09 = y2009[25:50]
top_2550_19 = y2019[25:50]

rest_89 = y1989[50:]
rest_99 = y1999[50:]
rest_09 = y2009[50:]
rest_19 = y2019[50:]

# Create data for stacked bars
emissions_89 = [sum(top_10_89['Total']), sum(top_1025['Total']), sum(top_2550['Total']), sum(rest_89['Total'])]
emissions_99 = [sum(top_10_99['Total']), sum(top_1025_99['Total']), sum(top_2550_99['Total']), sum(rest_99['Total'])]
emissions_09 = [sum(top_10_09['Total']), sum(top_1025_09['Total']), sum(top_2550_09['Total']), sum(rest_09['Total'])]
emissions_19 = [sum(top_10_19['Total']), sum(top_1025_19['Total']), sum(top_2550_19['Total']), sum(rest_19['Total'])]

# Create labels for the x-axis and each stack
labels = ['Top 10', '11-25', '26-50', 'Rest']

# Set up the figure and axes
fig, ax = plt.subplots()

# Set up the positions and width of each bar
bar_positions = np.arange(len(labels))
bar_width = 0.15

# Create the stacked bars for each year
ax.bar(bar_positions - bar_width, emissions_89, bar_width, label='1989')
ax.bar(bar_positions, emissions_99, bar_width, label='1999')
ax.bar(bar_positions + bar_width, emissions_09, bar_width, label='2009')
ax.bar(bar_positions + 2*bar_width, emissions_19, bar_width, label='2019')

# Add labels and title
ax.set_xlabel('GDP Per Capita Ranking')
ax.set_ylabel('Total Emissions (million metric tons of CO2)')
ax.set_title('Total Emissions by GDP Per Capita Quartiles for Four Years')
ax.set_xticks(bar_positions)
ax.set_xticklabels(labels)
ax.legend()

# Show the plot
plt.show()


# It appears that the inequality of emissions discussed earlier is currently decreasing, as poorer countries contribute more to the total emissions. This trend aligns with Schipper, Ting et al.'s 1997 study, which explored the decrease in energy use per capita in developed nations (Schipper, L., Ting, M., Khrushch, M., Unander, F., Monahan, P.A., & Golove, W. (1996)). The evolution of carbon dioxide emissions from energy use in industrialized countries: an end-use analysis). This trend continued well into the 90s, despite increased growth in GDP. The reasons for this trend include the ability of industrialized nations to transition more easily into renewable energy source development, as well as growing consciousness around the issue.
# 
# Nevertheless, this may also be due to other unrelated factors, such as the rise in per capita wealth of micro-states, both in europe (such as Lichtenstein, Monaco, Luxembourg) as well as in Asia (Qatar, UAE, Bahrain) that although very wealthy, have low total emissions because of their size.

# In[45]:


emerged_df.dropna(inplace=True)
emerged_df = emerged_df.replace(',', '', regex=True)

# Convert 'GDP Per Capita' column to numeric type
emerged_df['GDP Per Capita'] = pd.to_numeric(emerged_df['GDP Per Capita'], errors='coerce')

# Remove any rows with missing values in 'GDP Per Capita' column
emerged_df.dropna(subset=['GDP Per Capita'], inplace=True)

x = emerged_df['GDP Per Capita']
y = emerged_df['Per Capita']
xmin = 0
xmax= x.max()

fig, ax = plt.subplots()
plt.hexbin(x, y, gridsize=35, cmap="plasma", mincnt = 1)
ax.set_xlim(xmin, xmax)
# Set axis labels and title
ax.set_xlabel('GDP Per Capita')
ax.set_ylabel('Emissions Per Capita')
ax.set_title('GDP Per Capita and Emissions Per Capita')

plt.colorbar().set_label('Number of Observations')
plt.show()


# The hexbin plot above shows that the relationship between GDP per capita and emissions per capita is not very clear. A cluster of countries with low GDP and low emissions is visible in the lower left corner of the plot. As we move along the GDP per capita axis, we notice a significant increase in carbon emissions from the 20,000 to $70,000 range, followed by a steady decrease in per capita emissions afterwards. One possible explanation for this is that, in the earlier years when the data was being collected (in the 1980s and 1990s), nations were less energy efficient. Therefore, since the chart captures data from multiple years, the relationship between GDP per capita and per capita emissions may be blurred.
# 
# To explore this further, we created a similar graph for a single year and observed a similar relationship between GDP per capita and emissions per capita. However, the numbers were small enough for the relationship to be considered noise and not statistically significant. These charts serve mainly to illustrate the main message that GDP per capita has *some* effect on emissions per capita, but further statistical analysis is required to establish the magnitude of the effect, which will be performed in this report.

# In[46]:


emerged_df.dropna(inplace=True)
emerged_df = emerged_df.replace(',', '', regex=True)
df_2019 = emerged_df[emerged_df['Year'] == 2019]


x = df_2019['GDP Per Capita']
y = df_2019['Per Capita']
xmin = 0
xmax= x.max()

fig, ax = plt.subplots()
plt.hexbin(x, y, gridsize=35, cmap="plasma", mincnt = 1)
ax.set_xlim(xmin, xmax)
# Set axis labels and title
ax.set_xlabel('GDP Per Capita')
ax.set_ylabel('Emissions Per Capita')
ax.set_title('GDP Per Capita and Emisions Per Capita')

plt.colorbar().set_label('Number of Oberservations')
plt.show()


# ## 3.6 Adding another Dataset
# 
# Given our study's objective of determining whether a significant relationship exists between household size and carbon emissions, it's essential to introduce population density data in our regression analysis to control for its effects. This will enable us to isolate the effect of household size on carbon emissions and determine whether it's a significant factor or whether household size is serving as a proxy for other variables such as GDP per capita and/or density.
# 
# It's important to note that density is distinct from household size as it doesn't capture the housing unit situation or actual living conditions in a country. For instance, a country like Algeria may have a low population density due to the desert nature of its landscape, but may have a high household size due to dense living in arable land. Therefore, controlling for density in our analysis is critical to accurately assess the impact of household size on carbon emissions.

# In[47]:


# We're going to data scrape again, as it's easier than going through the UN's very thorough datasets

URL = 'https://en.wikipedia.org/wiki/List_of_countries_by_past_and_future_population_density'

# Connect to the URL
Response = requests.get(URL) 

# Parse HTML and save to BeautifulSoup object¶
largest_soup = BeautifulSoup(Response.text, "html.parser") #html.parser telling Beautiful Soup that it's an HTML file

data_Table = largest_soup.find_all('table', {'class': 'wikitable'})[0]
header_row = data_Table.find('tr')
columns = [col.get_text(strip=True) for col in header_row.find_all('th')]

rows = data_Table.find_all('tr')[1:]
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

pop_den = pd.DataFrame(data, columns=columns)
pop_den.rename(columns = {'Country (or area)': 'Country'}, inplace = True)
pop_den['Country'] = pop_den['Country'].str.replace('*', '')

pop_den


# In[48]:


Melted_df = pd.melt(pop_den, id_vars=['Country'], var_name='Year', value_name='Population Density')
Melted_df['Year'] = Melted_df['Year'].astype('int64')
Melted_df['Country']= Melted_df['Country'].astype('str')
Melted_df['Country']= Melted_df['Country'].str.strip()
Emerged_df = pd.merge(emerged_df, Melted_df, on=['Country', 'Year']) #creating dataset with all categories
Emerged_df = Emerged_df.replace(',', '', regex=True)


# In[49]:


Emerged_df


# In[50]:


Emerged_df["Population Density"] = Emerged_df["Population Density"].astype('float64')
Emerged_df


# In[51]:


Emerged_no_outliers = Emerged_df[Emerged_df["Population Density"] <= 400]


# In[52]:


# Preparation
X = Emerged_df['Population Density'].values.reshape(-1,1)
X = X.astype('float64')
y = Emerged_df['Per Capita'].values


# In[53]:


# Train
ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)


# In[54]:


# Evaluate 
r2 = model.score(X, y)


# In[55]:


plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='k', label='Regression model')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('Per capita emissions', fontsize=14)
ax.set_xlabel('Population Density', fontsize=14)
ax.text(0.8, 0.1, '$R^2= %.2f$' % r2, fontsize=18, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('OLS Regression of Per Capita Emissions on Population Density' )

fig.tight_layout()


# In[56]:


X = Emerged_no_outliers['Population Density'].values.reshape(-1,1)
X = X.astype('float64')
y = Emerged_no_outliers['Per Capita'].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)

r2 = model.score(X, y)

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='k', label='Regression model')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('Per capita emissions', fontsize=14)
ax.set_xlabel('Population Density', fontsize=14)
ax.text(0.8, 0.4, '$R^2= %.2f$' % r2, fontsize=18, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('OLS Regression of Per Capita Emissions on Population Density' )

fig.tight_layout()


# With and without presence of outliers that contribute to noisy results, the relationship between population density and per capita emissions only show R-squared values of around 0, indicating a weak correlation between the two variables' variances. This finding is surprising as one would assume that higher population density would result in more efficient resource use. One reason behind this weak relationship could be the different characteristics population density is picking up. It could be picking up a poorer country, with high fertility rates and low per capita emissions, or a highly urbanized state with high per capita emissions. Density itself depending on the place may have a positive or negative effect on emissions. As we'll see in Yonghong Liu's study, densification was linked with higher emissions, but in a North American situation, when standing opposed to subarbinazation, dense urbanization is actually more energy efficient (https://suburbs.info.yorku.ca/2014/02/suburbanization-and-density-a-few-critical-notes/).
# 
# To better understand the relationship between household size, GDP per capita, population density and per capita emissions we have included a correlation table in this stage of the paper. By controlling for other variables that may impact household size and carbon emissions, we hope to examine the effect of household size on emissions more effectively. Furthermore, this correlation matrix lays the groundwork for the upcoming multivariate regression analysis.

# In[57]:


EMERGED_df = pd.merge(Emerged_df, household_data, on=['Country', 'Year'])

EMERGED_df= EMERGED_df.replace(',', '', regex=True)


EMERGED_df['GDP Per Capita'] = EMERGED_df['GDP Per Capita'].astype('float')
EMERGED_df['Population Density'] = EMERGED_df['Population Density'].astype('float')
EMERGED_df['Average household size (number of members)'] = EMERGED_df['Average household size (number of members)'].astype('float')

EMERGED_df.rename(columns={'Per Capita': 'Per Capita Emissions'}, inplace=True)
EMERGED_df


# In[58]:


cols = ['Per Capita Emissions','GDP Per Capita', 'Population Density', 'Average household size (number of members)']


corr = EMERGED_df[cols].corr()  # calculate correlation matrix

mask = np.triu(np.ones_like(corr))

# plot the correlation matrix as a heatmap
sn.heatmap(corr, cmap='coolwarm', annot=True, mask = mask)

# set plot title
plt.title('Correlation Matrix')

# display the plot
plt.show()


# Population's purported link to carbon emissions may be due, in part, to its weak correlation with GDP per capita and the subsequent moderate effect GDP per capita has on per capita emissions. Because of this study’s a global scope, it does not directly contradict Yonghong Liu's 2017 paper ("The impact of urbanization on GHG emissions in China: The role of population density," Journal of Cleaner Production 157 (2017): 299-309), which found that density played a significant role in increasing China's carbon emissions. However, it challenges the possible assumption that this conclusion could be extrapolated and considered universal.
# 
# Another intriguing finding is the negative correlation between average household size and population density. One might expect that less space/more density would result in more people living in each housing unit, but the relationship may not be statistically significant once a regression is run. Nevertheless, it is noteworthy that there is no positive relationship between the two. One possible explanation is the pattern of high-density urbanization in developed countries, where smaller households are more common due to cultural reasons, as singles and couples live without members of their extended family. 
# 
# Below we have plotted the relationships between household size and both GDP per capita and population density to see how they affect per capita emissions. We have also calculated the r-squared of household size on emissions. Surprisingly, we found that both relationships had lower r-squared values compared to the previous regression of household size on per capita emissions (see "Per Capita Emissions and Household Size" plot). Adding more data usually improves the accuracy of the regression, so this result was unexpected. However, upon re-calculating the r-squared of household size on emissions, we found that it was now 0.26, as opposed to the previous value of 0.46. This suggests that the relationship between household size and emissions has been weakened in the merged dataset, possibly due to the presence of other important factors. The merged dataset contains more recent data, mostly due to scarce information on GDP per capita stretching significantly into the past.

# In[59]:


# Creating dataset
z = EMERGED_df['Per Capita Emissions'].values
x = EMERGED_df['Average household size (number of members)'].values
y = EMERGED_df['Population Density'].values

# Performing multiple linear regression
X = sm.add_constant(np.column_stack((x, y)))
model = sm.OLS(z, X).fit()

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

# Creating plot
ax.scatter3D(x, y, z, color='green')
plt.title('GDP Per Capita and Household Size on Per Capita Emissions')
ax.set_ylabel('Population Density')
ax.set_xlabel('Average Household Size')
ax.set_zlabel('Per Capita Emissions')

# Displaying R-squared
R_squared = round(model.rsquared, 2)
R_squared = str(R_squared)
ax.text2D(0.5, 0.5, 'R-squared = ' + R_squared, fontsize=12, transform=ax.transAxes)

# Show plot
plt.show()


# In[60]:



# Creating dataset
z = EMERGED_df['Per Capita Emissions'].values
x = EMERGED_df['Average household size (number of members)'].values
y = EMERGED_df['GDP Per Capita'].values

# Performing multiple linear regression
X = sm.add_constant(np.column_stack((x, y)))
model = sm.OLS(z, X).fit()

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d')

# Creating plot
ax.scatter3D(x, y, z, color='green')
plt.title('GDP Per Capita and Household Size on Per Capita Emissions')
ax.set_ylabel('GDP Per Capita')
ax.set_xlabel('Average Household Size')
ax.set_zlabel('Per Capita Emissions')

# Displaying R-squared
R_squared = round(model.rsquared, 2)
R_squared = str(R_squared)
ax.text2D(0.5, 0.5, 'R-squared = ' + R_squared, fontsize=12, transform=ax.transAxes)

# Show plot
plt.show()


# In[61]:


# x and y are your variables of interest
x = EMERGED_df['Average household size (number of members)'].values
y = EMERGED_df['Per Capita Emissions'].values

# Calculate the slope, intercept, correlation coefficient, p-value, and standard error of the regression coefficients
slope, intercept, rvalue, pvalue, stderr = linregress(x, y)

# Calculate R-squared value
r_squared = rvalue**2

print("R-squared:", r_squared)


# # OLS Regression
# 
# ## 4.1 
# At first glance, the relationship between household size and carbon emissions would seem to be linear; as more people cram into the same space there should be just more energy efficiency, even if we deal with the collinearity it has with GDP per capita. However, if we look at the household size and Per Capita emissions plot (figure XXX) we can see that there is a negative linear relationship, until the household size reaches 5 people; once it reaches this point the relationship disappears as the line of best fit becomes almost completely horizontal.
# 
# The reasons why this could be the case are numerous. For one, there may be a decreasing marginal efficiency that one achieves within the household. For example, with four people in the household instead of two, the same appliances can be used but at a higher capacity rate. With more than five people, these inefficiencies may dissipate as families may have to buy more appliances and use some of them at similar capacity rates as a two-person household.
# 
# There's also factors external to the household that are influenced by holding high household sizes. First off, more than five people in a household may be indicative of bad urban planning whereby communities are dense but also chaotic, leading to higher per capita carbon emissions as energy, transportation and resources are not allocated efficiently. This can be tied back to Jevon's paradox, where an increase in the efficiency of resource use leads to an increase in that resource's consumption rather than a decrease. In this case, the efficiencies brought by denser households lead to an increase in energy consumption and a leveling off of the negative effect density had on carbon emissions.
# 
# ## 4.2
# For the X's we choose for the following regressions, they will mostly consist of variables that have already been mentioned. Earlier in the report, oil emissions were singled out as a potential substitute for coal, and could yield an accurate regression with significant effects on per capita emissions. Other emissions will also be used for this purpose, depending on their source, which may come from low-carbon renewable sources like solar panel or wind turbine production, or higher carbon sources like biomass. A regression would help clarify this relationship.
# 
# In addition, household size is the main variable being studied in this paper. As explained previously, the effect of household size on emissions could depend on a multitude of factors such as energy efficiency in the region, type of building, as well as its collinearity with density and GDP per capita. For this reason, these two variables will be included as fixed effects and x variables, to determine the effect of these important factors on emissions.

# Justification of why run the regression and why run them as an OLS

# In[102]:


x = emissions_data_clean[['const','Coal', 'Oil', 'Cement', 'Flaring', 'Other', 'Gas']]

emissions_data_clean['const'] = 1
reg1 = sm.OLS(endog=emissions_data_clean['Per Capita'], exog= x,
             missing = 'drop')
results = reg1.fit()
results.summary()


# In[106]:


results.mse_resid


# ### Regression Evaluation
# The regression has an R-squared of 0.002 which is lower than all the previous regressions run and means that very little of the variation of per capita emissions can be explained through the variation in the different emission sources. Although it must be measured against other  models to determine its validity, this regressions AIC is also extremely high, meaning the model does not have a great fit or may be overparameterized; This weakens any conclusions that can be brought out from the regression table.
# 
# ### Regression Takeaways
# Despite its glaring weaknesses, it is very interesting to note that all the emission sources have significant effects on per capita emissions, except for "other," which deserves more exploration. Coal, flaring, and gas had the expected positive effects on per capita emissions, but oil and cement had negative effects. The negative effect of oil may help substantiate our hypothesis regarding why oil was chosen as an explanatory variable. The negative effect of cement on emissions is certainly counter-intuitive, but it may be due to collinearity between concrete production and a country's economic status. Developing countries tend to produce more concrete, while developed countries' concrete usage tapers off as they reach economic maturity (Rosamund Pearce, “Q&A: Why Cement Emissions Matter for Climate Change - Carbon Brief,” Carbon Brief, September 13, 2018). Due to these collinearities and omitted variable bias, these results are difficult to interpret, but they nevertheless give us some direction as to what should be further examined.

# # Machine Learning
# In this section we'll use a machine learning algorithm to determine, first the effects each source has on total emissions and then the effects specific characteristics of urban planning have on emissions. We're in theory, looking at and attempting to quantify the same things  we examined in the previous linear regression section, however, now we're able to quantify these effects in non-linear ways because of the way the decision tree functions, where each internal node represents a decision based on a predictor variable and each leaf node represents the predicted value of the response variable.
# 
# In this case, there's no reason to suggest that, for example, the relationship between oil emissions and total emissions must be linear. It could be hypothesized that small number of oil emissions could decrease total emissions as it represents a shift from the more pollutnt coal emissions, but that as the amount of oil emissions increases it could also represent a growing dependence on the energy source at the expense of low carbon, renewable energy sources. This type of reasoning could exist for multiple variables (Density for example) and highlights the need for a model that can map more complex relationships.
# 
# For the first regression tree the model will be the same as the one previously used for the linear regression
# 
# $$
# Total Emissions = \beta_0 + \beta_1 \text{Coal}+ \beta_2 \text{Oil} + \beta_3 \text{Gas} + \beta_4 \text{Cement} + \beta_4 \text{Flaring} + \beta_5 \text{Other} + \epsilon
# $$
# 
# 

# In[71]:


X = emissions_data_clean.drop(["Country", "ISO 3166-1 alpha-3", "Year", "Total", "Per Capita"], axis=1).copy()
y = emissions_data_clean["Total"]

Emissions_tree = tree.DecisionTreeRegressor(max_depth = 4).fit(X, y)
y_pred_tree = Emissions_tree.predict(X)
print ("Mean Squared Error:", metrics.mean_squared_error(y, y_pred_tree))


# In[84]:


Emissions_figure = plt.figure(figsize=(50, 50))
Emissions_figure = tree.plot_tree(Emissions_tree, feature_names=X.columns, filled=True)


# One of the first results we can see from the regression tree is that whether oil emissions are equal or below 5303.686 metric tons per capita, is the most important decision that a country can make. In other words, the machine learning algorithm has decided that the first node that would reduce the mean squared error the most would be the oil emissions at or below that quantity. This already goes on to reinforce our beliefs surrounding the importance of oil emissions in dicating total emissions, as well as thier potential non-linear effects.
# 
# Lastly, it is important to note the absence of "other" emissions in the table. In none of the 15 nodes is it mentioned, meaning it most likely has a small effect on total emissions, most likely due to their small overall emissions, or they are not sufficiently correlated with a certain range of emissions of another emissions source.
# 
# Ultimately, the fact that the squared error of many of the nodes, including 7 out of the 8 final nodes), are significantly larger than the value they are predicting, means that the model is very inaccurate and calls for a better method of analysis.

# ## Bootstrap Aggregation
# Below we will seek to fit multiple trees onto the bootstrapped data in order to come up with a more accurate model with a smaller MSE

# In[99]:


regr1 = RandomForestRegressor(max_features= 6, random_state=1) 
regr1.fit(X, y)

pred = regr1.predict(X)

plt.scatter(pred, y, label='Total Emissions')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y')


mean_squared_error(y, pred)


# Here we constructed a more accurate regression that yields an MSE of around 136, a figure much lower than 17807 that was found using one regression tree and even lower than the 267 yielded by the OLS regression. Visually we notice that it maps quite well to the data and gives legitimacy to this model.

# In[100]:


Importance = pd.DataFrame({'Importance':regr1.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='black', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# Using this new model we are able to confirm the predictions we made using the regression tree, namely that Oil is crucial in predicting total emissions. The X axis, variable importance displays the percentage increase in the model's accuracy by the inclusion of the variable. 

# In[115]:


rf = RandomForestRegressor(n_estimators=1, max_depth=5, random_state=1)

x = emissions_data_clean['Oil']
rf.fit(x.values.reshape(-1, 1), y)
ice_df = ice(emissions_data_clean, 'Oil', rf.predict)

fig, ax = plt.subplots()
ice_plot(ice_df, c='black', alpha=.15, linewidth=.5)
plt.xlabel('Oil')
plt.ylabel('Emissions')


# ## Conclusion and Next Steps
# This paper builds upon the original previous research by incorporating the new United Nations Data (United Nations, Department of Economic and Social Affairs, Population Division (2022).Database on Household Size and Composition 2022. UN DESA/POP/2022/DC/NO. 8.) and testing for the relationship between household size and per capita emissions. The scatter plot at the beginning illustrated a slight negative relationship between household size and per capita emissions in the 2-4 person range, without taking fixed effects or endogeneity into account.
# 
# The presented maps showcased the distribution of both household sizes and coal emissions. The inclusion of coal emissions allowed for further exploration of Balmer's 2017 South Africa study and the identification of potential macro-level implications of the observed relationship between low-income, large household sizes, and high coal usage.
# 
# We also added data GDP per Capita data from the IMF (https://www.imf.org/en/Publications/WEO/weo-database/2022/October) as well as population density data from the UN (https://www.un-ilibrary.org/content/books/9789210001014), both scrapped from Wikipedia for logistical reasons. Both were included to rule out confounding bias, as household size may be strongly linked with these two other variables. 
# 
# By using actual GDP per capita instead of G7 status to proxy for wealth, this edition was able to establish a more robust relationship between wealth, per capita emissions, and other variables. Our analysis of total emissions by rank in per capita emissions showed that wealthier nations are actually decreasing their overall contributions to total emissions relative to the rest of the world, suggesting that other factors may be important in the production of carbon emissions. We also examined the relationship between population density and carbon emissions through a linear regression and found a low R-squared, indicating a weak link between these two variables.
# 
# Finally, we created a correlation table to lay the groundwork for future multivariate regression analysis and found that household size and GDP per capita had the strongest effects on emissions. However, these variables are negatively correlated, which underscores the need for further analysis.
# 
# As mentioned before, further research should include a multivariate regression analysis to determine the statistical significance of each explanatory variable's effect on emissions per capita, while controlling for the other variables. In addition, exploration of the relationship between household size and density is recommended because of the conclusions that run counter to common assumptions and Yonghon Liu's 2017 paper. A qualitative study of why household size may also be required in order to offer more concrete suggestions to governments and urban planning organizations.
