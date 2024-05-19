import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataframe = pandas.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

label_encoder = LabelEncoder()
dataframe["Gender"] = label_encoder.fit_transform(list(dataframe["Gender"]))
dataframe["family_history_with_overweight"] = label_encoder.fit_transform(list(dataframe["family_history_with_overweight"]))
dataframe["FAVC"] = label_encoder.fit_transform(list(dataframe["FAVC"]))
dataframe["CAEC"] = label_encoder.fit_transform(list(dataframe["CAEC"]))
dataframe["SMOKE"] = label_encoder.fit_transform(list(dataframe["SMOKE"]))
dataframe["SCC"] = label_encoder.fit_transform(list(dataframe["SCC"]))
dataframe["CALC"] = label_encoder.fit_transform(list(dataframe["CALC"]))
dataframe["MTRANS"] = label_encoder.fit_transform(list(dataframe["MTRANS"]))
dataframe["NObeyesdad"] = label_encoder.fit_transform(list(dataframe["NObeyesdad"]))

dataframe["Weight"] = dataframe["Weight"] * 2.20462

X = dataframe.drop(columns=["Weight"])
y = dataframe["Weight"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
accuracy = random_forest.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
