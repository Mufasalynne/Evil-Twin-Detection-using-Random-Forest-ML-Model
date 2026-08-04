import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import joblib


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
#print(data["SSID"].value_counts())
#print(data["Vendor_OUI"].value_counts())
#print(data["Security_Type"].value_counts())

# Feature Engineering 
x = data[["SSID", "RSSI", "Channel", "Security_Type","Vendor_OUI","Beacon_Interval_ms", "Duplicate_SSID_Count_30min"]]
y = data["Label"]

# feature encoding for categorical features
x = pd.get_dummies(x, columns=["SSID", "Vendor_OUI", "Security_Type"], drop_first= True)

# encode the label_data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(x.info())
print(x.head(10))

print(y[:10])

# Spliting the dataset into the training and the testing dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

# train the model using random forest classifier
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(x_train, y_train)

# predicting the test dataset
y_pred = rf.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy:.2f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#classification report
class_report = classification_report(y_test, y_pred)
print(class_report)

# Feature Importance
importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by = "Importance",
    ascending = False
)

print(importance)

# ploting the results of the feature importance

importance.plot(
    x="Feature",
    y="Importance",
    kind="bar",
    legend=False

)
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

# Display the confusion matrix
ConfusionMatrixDisplay.from_estimator(rf, x_test, y_test)
plt.show()


# save model
joblib.dump(rf,"evil_twin_random_forest.pkl")
