import pandas as pd

# load data ;
data_path = "Evil_Twin_Example_Dataset.xlsx"

data = pd.read_excel(data_path)
print(data)

print(data.head(9))
print(data.info())

#print the missing values;
print(data.isnull().sum())