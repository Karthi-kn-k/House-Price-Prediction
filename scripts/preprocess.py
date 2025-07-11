import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    data = pd.read_csv(path)
    
    # Drop non-numeric or irrelevant columns
    data = data.drop(columns=["id", "date", "zipcode"], errors='ignore')
    
    X = data.drop(columns=["price"])
    y = data["price"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Drop columns with >10% missing
    x_train = x_train.dropna(axis=1, thresh=len(x_train) * 0.9)
    x_test = x_test[x_train.columns]

    # Drop remaining NaNs
    x_train = x_train.dropna()
    y_train = y_train.loc[x_train.index]

    x_test = x_test.dropna()
    y_test = y_test.loc[x_test.index]

    # Standard Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
