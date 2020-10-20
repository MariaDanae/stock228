import configparser
import logging

import joblib
from sklearn.model_selection import train_test_split

from stock228.src.IO.get_data_from_yahoo import get_last_stock_price
from stock228.src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket
from stock228.src.algo.dummy_model import Stock_model


def create_business_logic():
    data_fetcher = get_last_stock_price
    return BusinessLogic(Stock_model(data_fetcher))


class BusinessLogic:

    def __init__(self, model_creator):
        self._root_bucket = 'model_bucket_ycng_228_maria'
        self._config = configparser.ConfigParser()
        self._config.read('application.conf')
        self._model_creator = model_creator
        self._create_bucket()

    def get_version(self):
        return self._config['DEFAULT']['version']

    def get_bucket_name(self):
        return f'{self._root_bucket}_{self.get_version().replace(".", "")}'

    def _get_or_create_model(self, ticker):
        log = logging.getLogger()
        model_filename = self.get_model_filename_from_ticker(ticker)
        model = get_model_from_bucket(model_filename, self.get_bucket_name())
        if model is None:
            log.warning(f'training model for {ticker}')
            model = self._model_creator.fit(ticker)
            with open(model_filename, 'wb') as f:
                joblib.dump(model, f)
            upload_file_to_bucket(model_filename, self.get_bucket_name())
        return model

    def get_model_filename_from_ticker(self, ticker):
        return f'{ticker}.pkl'

    def _create_bucket(self):
        create_bucket(self.get_bucket_name())

    def do_predictions_for(self, ticker):
        model = self._get_or_create_model(ticker)
        predictions = model.predict(ticker)
        return predictions

    # def create_df(self, ticker):
    #     model = self._get_or_create_model(ticker)
    #     return model.create_df(ticker)
    #
    # def create_X_Y_test(self, df_lags, test_size=0.2):
    #     X = df_lags.drop('lags_0', axis=1)
    #     Y = df_lags[['lags_0']]
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    #     return X_train, X_test, y_train, y_test