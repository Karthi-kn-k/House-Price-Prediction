from sklearn.linear_model import LinearRegression

def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

