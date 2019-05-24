import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import urllib
import re
import json
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import classification_report, average_precision_score, mean_squared_error
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

def get_currency_rate(date="2016-01-01"):
    """ Fetches currency rates (to USD) for a given date. All currency types available at xe.com are included."""
    
    request = urllib.request.urlopen("https://www.xe.com/currencytables/?from=USD&date="+date)
    raw_data = request.read().decode("utf8")
    request.close()
    pattern = """<tr><td><a href='(?P<url>.*?)'>(?P<CURRENCY>.*?)</a></td><td>(?P<CURRENCY_LONG>.*?)</td>"""+\
        """<td class="historicalRateTable-rateHeader">(?P<UNITS_PER_USD>.*?)</td>"""+\
        """<td class="historicalRateTable-rateHeader">(?P<USD_PER_UNIT>[0-9.]+?)</td></tr>"""
    compiled_pattern = re.compile(pattern)
    
    currency_table = []

    for currency_rate in compiled_pattern.finditer(raw_data):
        currency_rate_dict = currency_rate.groupdict()
        currency_table.append({'currency': currency_rate_dict['CURRENCY'],
                            'usd_per_unit': float(currency_rate_dict['USD_PER_UNIT'])})

    df_currency_rates = pd.DataFrame.from_dict(currency_table)
    df_currency_rates = df_currency_rates.loc[df_currency_rates['currency'].drop_duplicates().index, :]
    df_currency_rates = df_currency_rates.set_index('currency', drop=False)

    return df_currency_rates

def get_cryptocurrency_rate(currency, timestamp):
    """Fetches cryptocurrency rates (to USD) for a given currency and timestamps """
    
    api_key = '8a79cd82ccdf05f9353dd3b5fb76e08f1f4ed0e13873d2de6aa606eadcfbd9f1'
    url = "https://min-api.cryptocompare.com/data/pricehistorical?fsym="+currency+"&tsyms=USD&ts="+timestamp+"&api_key="+api_key
    request = urllib.request.urlopen(url)
    raw_data = request.read().decode("utf8")
    
    return json.loads(raw_data)[currency]['USD']

def convert_country_code_to_name(df_countries):
    """Helper method to convert country code to long country name"""
    def _country_code_to_name(code):
        try:
            # Trim country name so that charts will look better
            return str(df_countries.loc[code, 'NAME'])[:15]
        except:
            return None

    return _country_code_to_name

def get_fraudster_label_by_user_id(df_users):
    """Helper method to get fraudster label by user id"""
    def _get_fraudster_label_by_user_id(x):
        try:
            return df_users.loc[x, 'IS_FRAUDSTER']
        except:
            return None

    return _get_fraudster_label_by_user_id

def get_usd_per_unit():
    """Helper method to get learn USD per unit for given transaction. Useful for comparing transactions with varying currency types."""
    
    cache_currency_rates = {}

    def _get_usd_per_unit(row):
        try:
            timestamp = row['created_date'].to_period('M').to_timestamp()
            date = timestamp.isoformat().split('T')[0]
            if date not in cache_currency_rates:
                cache_currency_rates[date] = get_currency_rate(date)

            df_currency_rates = cache_currency_rates[date]

            return df_currency_rates.loc[row['currency'], 'usd_per_unit']
        except Exception as e:
            return None
            ## It is crypto currency, we will fetch usd_per_unit data from a different api. In addition, we are going to
            ## fetch currency rates at the nearest first day of the month given a transaction timestamp
            timestamp = int(row['created_date'].to_period('M').to_timestamp().timestamp())
            
            cache_hash = row['currency']+str(timestamp)
            
            if cache_hash not in cache_currency_rates:
                cache_currency_rates[cache_hash] = get_cryptocurrency_rate(row['currency'], str(timestamp))
            
            return cache_currency_rates[cache_hash]
    
    return _get_usd_per_unit

def load_transformed_tables_from_csv():
    ## I assume that currency rates (not cryptocurrency) don't vary significantly over time so I fetched the data of 2016-01-01
    ## I also use USD as reference currency and convert all the currencies to USD to compare amounts of transactions fairly.
    df_currency_rates = get_currency_rate()

    df_countries = pd.read_csv('data/countries.csv').set_index('CODE', drop=False)
    df_countries_long_index = df_countries.set_index('CODE3', drop=False)
    df_currency_details = pd.read_csv('data/currency_details.csv').set_index('CCY', drop=False)
    df_transactions = pd.read_csv('data/transactions.csv')
    df_users = pd.read_csv('data/users.csv').set_index('ID', drop=False)

    df_transactions.loc[:, 'CREATED_DATE'] = df_transactions['CREATED_DATE'].astype('datetime64[ns]')
    df_transactions.loc[:, 'USD_PER_UNIT'] = df_transactions.apply(get_usd_per_unit(df_currency_rates), axis=1)
    df_transactions.loc[:, 'IS_CRYPTO'] = df_transactions['CURRENCY'].apply(lambda x: df_currency_details.loc[x, 'IS_CRYPTO'])
    df_transactions.loc[:, 'EXPONENT'] = df_transactions['CURRENCY'].apply(lambda x: df_currency_details.loc[x, 'EXPONENT'])
    df_transactions['AMOUNT'] * df_transactions['USD_PER_UNIT'] * \
                                np.power(10.0,-1*df_transactions['EXPONENT'])

    df_users.loc[:, 'COUNTRY_NAME'] = df_users['COUNTRY'].apply(convert_country_code_to_name(df_countries))

    transaction_count_by_user = df_transactions.groupby('USER_ID')['AMOUNT'].count().rename('TRANSACTION_COUNT')
    transaction_count_by_user = transaction_count_by_user.loc[transaction_count_by_user.index.intersection(df_users.index)]

    df_transactions.loc[:, 'MERCHANT_COUNTRY_NAME'] = df_transactions.loc[:, 'MERCHANT_COUNTRY'].apply(
                                        convert_country_code_to_name(df_countries_long_index))

    df_users = pd.concat([df_users, transaction_count_by_user], axis=1)

    mean_transaction_period = df_transactions.groupby('USER_ID')['CREATED_DATE'].apply(lambda x: x.diff().abs().mean())
    mean_transaction_period = mean_transaction_period.astype('timedelta64[h]').rename('MEAN_TRANSACTION_PERIOD')

    df_users = pd.concat([df_users, mean_transaction_period], axis=1)

    df_transactions.loc[:, 'IS_FRAUDSTER'] = df_transactions['USER_ID'].apply(get_fraudster_label_by_user_id(df_users))

    ## Combine transactions and users table by using summary statistics (sum and median)

    usd_total_amount_by_user = df_transactions.groupby('USER_ID')['USD_AMOUNT'].sum().rename('USD_TOTAL_AMOUNT').astype(np.float32)
    usd_total_amount_by_user = usd_total_amount_by_user.loc[usd_total_amount_by_user.index.intersection(df_users.index)]

    usd_median_amount_by_user = df_transactions.groupby('USER_ID')['USD_AMOUNT'].median().rename('USD_MEDIAN_AMOUNT').astype(np.float32)
    usd_median_amount_by_user = usd_median_amount_by_user.loc[usd_median_amount_by_user.index.intersection(df_users.index)]
    df_users = pd.concat([df_users, usd_total_amount_by_user, usd_median_amount_by_user], axis=1)

    ## Remove 99th quantile of USD_TOTAL_AMOUNT
    quantile_99th = df_users["USD_TOTAL_AMOUNT"].quantile(0.99)

    df_users = df_users[df_users['USD_TOTAL_AMOUNT'] < quantile_99th]

    return df_users, df_transactions

def merge_small_columns(series, less_than=0.005):
    """ Merge small columns into OTHER. Useful for plotting readable pie charts. """
    
    other_cols = series[series < series.sum()*less_than].index
    if len(other_cols) > 1:
        series = series.append(pd.Series({'OTHER':series.loc[other_cols].sum()}))
        series.drop(other_cols, inplace=True)
    
    return series

def get_metric_results(clf, X, y, one_class=False, sample_weight=None):
    """
    Evaluates the prediction of a given model on given dataset with labels.
    :param clf: estimator object implementing ‘predict’
    :param X: np.ndarray or pd.DataFrame, features that will be used for prediction
    :param y: list, true labels of the samples given in X
    :param one_class: whether algorithm is one class type of algorithm or not.
    :param sample_weight: array-like of shape = [n_samples]
    :return: dictionary that stores the metric results.
    """
    results = {'roc_auc':None, 'avg_precision': None}
    
    y_predictions = clf.predict(X)

    if 'predict_proba' in dir(clf):
        probas = clf.predict_proba(X)
        results['roc_auc'] = roc_auc_score(y, probas[:, 1], sample_weight=sample_weight)
        results['avg_precision'] = average_precision_score(y, probas[:, 1], sample_weight=sample_weight)

    results['matthews'] = matthews_corrcoef(y, y_predictions, sample_weight=sample_weight)
    results['mse'] = mean_squared_error(y, y_predictions, sample_weight=sample_weight)
    results['f1_score'] = f1_score(y, y_predictions, sample_weight=sample_weight)

    return results


def plot_confusion_matrix(fig, confusion_matrix_, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param fig: pyplot.Figure instance
    :param confusion_matrix_: confusion matrix
    :param classes: list of class names
    :param normalize: bool, whether normalize the confusion matrix or not
    :param title: The title of the plot
    :param cmap: Colormap instance
    """
    if normalize:
        confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ax = fig.add_subplot(111)
    cax = fig.add_axes([0.83, 0.3, 0.08, 0.5])

    #print(confusion_matrix_)

    im  = ax.imshow(confusion_matrix_, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, cax=cax, ax=ax, fraction=0.2)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = confusion_matrix_.max() / 2.
    for i in range(confusion_matrix_.shape[0]):
        for j in range(confusion_matrix_.shape[1]):
            ax.text(j, i, format(confusion_matrix_[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if confusion_matrix_[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

def fill_missing_amount_usd(df_transactions):
    rows_with_missing_amount_usd = df_transactions['amount_usd'].isna()
    usd_per_unit = df_transactions.loc[rows_with_missing_amount_usd].apply(get_usd_per_unit(), axis=1)

    df_transactions.loc[rows_with_missing_amount_usd, 'amount_usd'] = df_transactions.loc[
                                                                          rows_with_missing_amount_usd, 'amount'] \
                                                                      * usd_per_unit * np.power(10.0,
                                                                                                -1 * df_transactions[
                                                                                                    'exponent'])
    return df_transactions

