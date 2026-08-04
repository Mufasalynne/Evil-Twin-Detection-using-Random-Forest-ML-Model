import pandas as pd
import matplotlib.pyplot as plt

# load data ;
data_path = "Evil_Twin_Example_Dataset.xlsx"

data = pd.read_excel(data_path)
print(data)

print(data.head(9))
print(data.info())


#CLEANING DATA
# Checking for dublicate data 

dublicate_count = data.duplicated().sum()
print(dublicate_count)

# drop the duplicated data 
data = data.drop_duplicates()
print(data)

#HANDLING MISSING VALUES
missing_data = data.isnull().sum()
missing_data = missing_data[missing_data > 0]
print(missing_data)

# drop the missing Values 
drop_missing = data.dropna()
print(drop_missing)

# remove unconfired from verification_method column

#Exploratory Data Analysis (EDA)
print(data["Label"].value_counts())

print(data["RSSI"].describe())

plt.hist(data["RSSI"], bins=20)
plt.xlabel("RSSI")
plt.ylabel("frequency")
plt.show()


print(data["Channel"].value_counts())
print(data["SSID"].value_counts())
print(data["Vendor_OUI"].value_counts())
print(data["Security_Type"].value_counts())

# Feature Engineering 
#x = data[["SSID", "RSSI", "Channel", "Security_Type", "Beacon_Interval_ms", "Duplicate_SSID_Count_30min"]]
#y = data["Label"]

