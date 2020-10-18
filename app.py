from flask import Flask

from stock228.src.IO.get_data_from_yahoo import get_last_close_price
from stock228.src.business_logic.process_query import create_business_logic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Hello you should use an other route:!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction = bl.do_predictions_for(ticker)
    last_close_price = get_last_close_price(ticker)
    last_close = last_close_price.get("close")

    return f'Prediction for ticker {ticker} {prediction}' \
           f'\n\nLast close price was: {last_close}'


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=5000, debug=True)
