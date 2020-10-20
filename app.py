from flask import Flask

from stock228.src.IO.get_data_from_yahoo import get_last_close_price, get_last_stock_price
from stock228.src.business_logic.process_query import create_business_logic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Hello you should use an other route:!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    # bl = create_business_logic()
    # df = bl.create_df(ticker)
    # X_train, X_test, y_train, y_test = bl.create_X_Y_test(df, test_size=test_size)
    # prediction_and_score = bl.do_predictions_for(X_train, X_test, y_test)
    # last_close_price = get_last_close_price(ticker)

    bl = create_business_logic()
    prediction_and_score = bl.do_predictions_for(ticker)
    last_close_price = get_last_close_price(ticker)

    result = f'Prediction for ticker {ticker}:   {prediction_and_score.get("prediction_for_y_train")}' \
             f'\n\nScore given with test_size 0.2 is:  ' \
             f'r2_score: {prediction_and_score.get("r2_score")}' \
             f'mean_absolute_error": {prediction_and_score.get("mean_absolute_error")}' \
             f'"mean_squared_error": {prediction_and_score.get("mean_squared_error")}' \
             f'"sqrt_mean_squared_error": {prediction_and_score.get("sqrt_mean_squared_error")}'\
             f'\n\nLast close price was: {last_close_price.get("close")}'

    return result



if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=5000, debug=True)
