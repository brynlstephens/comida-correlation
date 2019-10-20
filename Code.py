import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

from matplotlib import pyplot as plt
import numpy as np
from datascience import *

df = pd.read_csv('just tacos and burritos2.csv',error_bad_lines=False)
non_null_columns = [col for col in df.columns if df.loc[:, col].notna().any()]

new_df=df[non_null_columns]
new_df
df2 = new_df.set_index("postalCode", drop =True)
grouped_df=df2.groupby("postalCode").count()
new_grouped=grouped_df["id"]

his_df=pd.read_csv("hispPopData.csv")

merged_data= pd.merge( new_grouped,his_df, left_on='postalCode', right_on='GEO.id2')
merged_data

y=merged_data["id"]
x=merged_data["HC03_VC93"]

plt.scatter(x,y)
plt.ylim(0,100)

from sklearn.linear_model import LinearRegression

for i in x:
    i = float(i)

model = LinearRegression().fit(x.values.reshape(-1,1),y)
yhat = model.predict(x.values.reshape(-1,1))
plt.scatter(x,y)
plt.ylim(0,100)
plt.plot(x,yhat, 'r', label='Model: $\hat{{y}}$={:.4f}*x+{:.4f}'.format(model.coef_[0], model.intercept_))