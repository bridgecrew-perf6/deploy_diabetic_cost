
'''
This is a simple linear regression model to predit the CO2 emmission from cars
Dataset:
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions
for new light-duty vehicles for retail sale in Canada
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("dataset.csv")

# take a look at the dataset
#df.head()

#use required features

cdf = df[['INSULIN','PRISK','CC_DISEASE_CNT','CC_CHF_DAYS','CC_DM_COMP_DAYS','CC_DM_WO_COMP_DAYS','MORB_OBESE_IND','OBESITY_IND','COST']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :8]
y = cdf.iloc[:, -1]


regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''