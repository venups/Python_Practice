##########################################################################################
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
##########################################################################################

##########################################################################################
print('\nThe Date Incubator Data Challenge, October 2022')
print('Challenge Solutions by: Yojana Kamabathula')
print('Application for Part time Data Science Bootcamp, Winter 2023')

#Read Data from Calls for Service 2021
df = pd.read_csv('callsforsrvc2021.csv')
##########################################################################################

##########################################################################################
print('\nQuestion 1.1: How many rows are in the data set?')
n_rows = len(df.index)
print('Answer 1.1: ',n_rows)
#########################################################################################

#########################################################################################
print('\nQuestion 1.2: What fraction of calls deal with animals?')
animals_ratio = df['FINAL_CALL_TYPE'].str.contains('animals',case=False).sum()/n_rows
print('Answer 1.2:', "{:.9f}".format(animals_ratio))
#########################################################################################

#########################################################################################
print('\nQuestion 1.3: What is the average number of calls received for each final radio code?(exclude radio code with calls less than 50)')
avg_calls_per_code = np.array([i for i in df.FINAL_RADIO_CODE.value_counts().to_list() if i>50]).mean()
print('Answer 1.3:', avg_calls_per_code)
#########################################################################################

#########################################################################################
print('\nQuestion 1.4: What is the Pearson Correlation Coefficient of the number of calls received each day with the daily average temperature?')
# Read Pheonix Temperature data from 2021
df2 = pd.read_csv('phoenix_temperature_data.csv')
df2['avg_temp'] = df2[['High F', 'Low F']].mean(axis=1)
df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.sort_values(by='date')

date_time = df['CALL_RECEIVED'].str.split(' ', 1, expand=True)
df['date'] = date_time[0]
df['time'] = date_time[1]
df['date'] = pd.to_datetime(df['date'])

data_calls = df.date.value_counts().to_frame().reset_index()
data_calls.columns = ['date','No_of_Calls']
data_calls['date']=pd.to_datetime(data_calls['date'])
data_calls = data_calls.sort_values(by='date')

r = stats.pearsonr(data_calls['No_of_Calls'], df2['avg_temp'])

print('Answer 1.4:', r[0])
#########################################################################################

#########################################################################################
print('\nQuestion 1.5: What is the increase in calls for each month using the line of best fit?(exclude calls in January and December)')
monthly_calls = data_calls.groupby(by=data_calls.date.dt.month)['No_of_Calls'].sum().to_frame()
monthly_calls = monthly_calls[1:11]
m, c = np.polyfit(range(1,11), monthly_calls, 1)
print('Answer 1.5:',m[0])
#########################################################################################

