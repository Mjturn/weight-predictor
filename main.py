import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataframe = pandas.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

non_numerical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
label_encoder = LabelEncoder()

for column in non_numerical_columns:
    dataframe[column] = label_encoder.fit_transform(list(dataframe[column]))

dataframe["Weight"] = dataframe["Weight"] * 2.20462

X = dataframe.drop(columns=["Weight"])
y = dataframe["Weight"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
accuracy = random_forest.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
