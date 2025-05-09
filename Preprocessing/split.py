
from sklearn.model_selection import train_test_split


def split_data(df):
    # Zielvariable festlegen       
    y = df['price_in_euro']
    X = df.drop(['price_in_euro'], axis=1)

    # Identifiziere numerische und kategoriale Spalten
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Datensatz in Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test , X,y, categorical_features , numeric_features

