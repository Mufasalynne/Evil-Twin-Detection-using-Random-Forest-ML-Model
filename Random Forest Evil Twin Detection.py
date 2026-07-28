import pandas as pd

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
data = data.isnull().sum()
print(data)

# drop the missing Values 
data = data.dropna()
print(data)

# remove unconfired from verification_method column

if "Verification_Method" in data.columns:
    before = len(data)
    df = df[~df["Verification_Method"].str.contains("Unconfirmed", case=False, na=False)]
    df = df.reset_index(drop=True)
    removed = before - len(df)
if removed:
    print(f"Removed {removed} row(s) with unconfirmed labels (untrustworthy ground truth).")
