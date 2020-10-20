from flask import Flask
from pandas import np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from stock228.src.IO.get_data_from_yahoo import get_last_close_price, get_last_stock_price
from stock228.src.business_logic.process_query import create_business_logic

app = Flask(__name__)

if __name__ == '__main__':
    ticker = 'AAA'
    test_size = 0.2

    bl = create_business_logic()
    data_fetcher = get_last_stock_price
    data = data_fetcher(ticker, last=True)
    data = data.drop('ticker', axis=1)
    data = data.round(2)
    X = data.drop('close', axis=1)
    Y = data[['close']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round(2)
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    sqrt_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    y_pred_from_train = regressor.predict(X_train)
    y_pred_from_train = y_pred_from_train.round(2)


    # last_close_price = get_last_close_price(ticker)

    print(f'Prediction for ticker {ticker}:   {y_pred_from_train}'
          f'\n\nScore given with test_size {test_size} is:'
          f'\n\tmean_absolute_error": {mean_absolute_error}'
          f'\n\tmean_squared_error: {mean_squared_error}'
          f'\n\tsqrt_mean_squared_error": {sqrt_mean_squared_error}'
          # f'\n\nLast close price was: {last_close_price.get("close")}'
          )