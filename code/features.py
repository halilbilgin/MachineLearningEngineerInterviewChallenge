import pandas as pd
from etl import get_transactions, get_users, load_config, load_features, get_engine
import psycopg2 as pg
from utils import fill_missing_amount_usd

def create_features(df_users, df_transactions):
    """ Create features from user and transaction tables. """

    processed_dataset = pd.DataFrame()

    one_hot_vector_features = ['currency', 'merchant_country', 'source', 'state']

    for feature in one_hot_vector_features:
        feature_onehot = pd.concat([pd.get_dummies(df_transactions[feature], prefix=feature), df_transactions['user_id']], axis=1)

        feature_onehot = feature_onehot.groupby('user_id').sum()

        ## Remove the dimensions that have very low variance.
        feature_onehot = feature_onehot.loc[:,(feature_onehot.sum(axis=0)/feature_onehot.values.sum() > 0.001)]
        feature_onehot = feature_onehot.apply(lambda x: x/x.sum(), axis=1)

        processed_dataset = pd.concat([processed_dataset, feature_onehot], axis=1)

    processed_dataset = pd.concat([df_users, processed_dataset], axis=1)
    processed_dataset.drop(['terms_version', 'created_date', 'id', 'country_name', 'country',
                           'phone_country', 'state'], axis=1, inplace=True)

    processed_dataset.loc[processed_dataset['kyc'] != 'PASSED', 'kyc'] = 0
    processed_dataset.loc[processed_dataset['kyc'] == 'PASSED', 'kyc'] = 1
    processed_dataset.loc[:, 'has_email'] = processed_dataset['has_email'].replace({False: 0, True: 1})

    processed_labels = processed_dataset['is_fraudster'].replace({False: -1, True: 1})

    processed_dataset.drop('is_fraudster', axis=1, inplace=True)
    
    return processed_dataset, processed_labels

if __name__ == '__main__':
    database_config = load_config('../misc/database_config.yaml')[0]

    engine = get_engine(database_config)

    df_transactions = get_transactions(engine)
    df_transactions = fill_missing_amount_usd(df_transactions)
    df_users = get_users(engine, df_transactions)

    processed_data, processed_labels = create_features(df_users, df_transactions)

    processed_data.to_csv('../data/processed_dataset.csv', index_label='user_id')
    processed_labels.to_csv('../data/processed_labels.csv', index_label='user_id')

    load_features(engine, processed_data, processed_labels)
