import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

def convert_categorical_to_integer_labels(dataframe):
    for col in dataframe.select_dtypes(include='object').columns:
        if col == 'SalePrice':
            continue
        dataframe[col] = dataframe[col].fillna('Missing')
        # use fit_transform() to convert categorical data to integer representations
        dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe

def train_model_return_error(X, y, max_leaf_nodes):
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)
    model = RandomForestRegressor(max_leaf_nodes, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

train_data = convert_categorical_to_integer_labels(train_data)
test_data = convert_categorical_to_integer_labels(test_data)

X = train_data.loc[:, train_data.columns != 'SalePrice']
y = train_data.SalePrice

# determine ideal number of leaf nodes
candidate_leaf_nodes = [5,10,25,50,100,200,500,700,1000]
candidate_leaf_nodes_mae_index = [] # instatiate index
print("Determining ideal number of leaf nodes...")
for n in tqdm(candidate_leaf_nodes):
    mae = train_model_return_error(X, y, n)
    candidate_leaf_nodes_mae_index.append([n, mae])

ideal_leaf_node_list = min(candidate_leaf_nodes_mae_index, key = lambda x: x[1])
ideal_leaf_nodes = ideal_leaf_node_list[0]
print(f"Done! Ideal leaf nodes: {ideal_leaf_nodes} (with MAE of {ideal_leaf_node_list[1]})")

# w/ ideal nodes, define model
model = RandomForestRegressor(max_leaf_nodes=ideal_leaf_nodes, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)
model.fit(X_train, y_train)

# get feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# select top 80 features
ideal_features = feature_importance_df.head(80)['feature'].values
X = train_data[ideal_features]

mae = train_model_return_error(X, y, ideal_leaf_nodes)

print(f"Selected features: {ideal_features[0:5]}...")

model.fit(X, y)
test_data_predictions = model.predict(test_data[ideal_features])
df = pd.DataFrame({'Id': range(1461, 1461 + len(test_data_predictions)),
                   'SalePrice': test_data_predictions})
df.to_csv('./test_predictions.csv', index=False)
print("Predictions written to ./test_predictions.csv")