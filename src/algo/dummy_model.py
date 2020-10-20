import logging

from pandas import np
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def create_features(df_stock, nlags=10):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    df = df.dropna(axis=0)
    return df


def create_X_Y(df_lags):
    X = df_lags.drop('lags_0', axis=1)
    Y = df_lags[['lags_0']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, y_train, y_test


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        X_train, X_test, y_train, y_test = create_X_Y(df_features)
        self.lr.fit(X_train, y_train)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data)
        df_features = df_features.round(2)
        X_train, X_test, y_train, y_test = create_X_Y(df_features)
        self.lr.fit(X_train, y_train)
        y_pred = self.lr.predict(X_test)
        y_pred = y_pred.round(2)
        score = self.lr.score(X_test, y_test)
        mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
        mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
        sqrt_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        predictions = self.lr.predict(X_train)
        predictions = predictions.round(2)
        prediction_for_y_train = predictions.flatten()[-1]

#         shape = df_features.shape()
#         #Classification
#         for i in range(1,shape[1]+1):
#             if df_features[i]:
#
#         df_features['buysell']

        return {"prediction_for_y_train": prediction_for_y_train,
                "r2_score": score,
                "mean_absolute_error": mean_absolute_error,
                "mean_squared_error": mean_squared_error,
                "sqrt_mean_squared_error": sqrt_mean_squared_error,
                }

    def create_df(self, ticker):
        data = self._data_fetcher(ticker, last=True)
        return create_features(data)