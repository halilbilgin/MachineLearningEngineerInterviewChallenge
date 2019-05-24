import psycopg2 as pg
import yaml
import pandas as pd
import argparse
from utils import *
from os import path
import numpy as np
from sqlalchemy import create_engine
import sqlalchemy
from io import StringIO
import csv

def psql_insert_copy(table, conn, keys, data_iter):
    """
    Fast insert to postgresql. Used in pandas.DataFrame.to_sql
    Got from https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

def get_engine(database_config):
    ## Needed for pandas to_sql command
    engine = create_engine('postgresql+psycopg2://' + database_config['user'] + ':' + database_config['password'] + \
                           '@' + database_config['host'] + ':' + str(database_config['port']) + '/' + database_config[
                               'dbname'], use_batch_mode=True)

    return engine

def load_config(config_path):
    with open(config_path) as schema_file:
        config = yaml.load(schema_file)
    return config

def create_tables(config: list, engine: sqlalchemy.engine.base.Engine):
    con = engine.connect()

    for table in config:
        name = table.get('name')
        schema = table.get('schema')
        ddl = f"""DROP TABLE IF EXISTS {name}"""
        con.execute(ddl)

        ddl = f"""CREATE TABLE {name} ({schema})"""
        con.execute(ddl)

def load_tables(engine: sqlalchemy.engine.base.Engine, config: list, data_path: str):

    for table in config:
        table_name = table.get('name')
        print(table_name)
        table_source = path.join(data_path, f"{table_name}.csv")

        df = pd.read_csv(table_source)
        df.columns = map(str.lower, df.columns)
        #df_reorder = df[table.get('columns')]  # rearrange column here
        df.to_sql(table_name, engine, index=False, if_exists='append', method=psql_insert_copy)

def etl(database_config, schema_path: str, data_path):
    engine = get_engine(database_config)

    if type(schema_path) == str:
        config = load_config(schema_path)
    else:
        config = schema_path

    create_tables(config=config, engine=engine)
    load_tables(engine=engine, config=config, data_path=data_path)

def get_transactions(engine: sqlalchemy.engine.base.Engine):

    ## THIS WAS BEFORE TRANSACTIONS TABLE UPDATED BY REVOLUT
    ## I assume that currency rates (not cryptocurrency) don't vary significantly over time so I fetched the data of 2016-01-01
    ## I also use USD as reference currency and convert all the currencies to USD to compare amounts of transactions fairly.
    #df_currency_rates = get_currency_rate()
    ## ---

    df_transactions = pd.read_sql("""SELECT t.*, c.name AS merchant_country_name, cd.is_crypto, cd.exponent,
                              CASE WHEN f.user_id IS NULL 
                                    THEN FALSE
                                    ELSE TRUE
                              END AS is_fraudster
                              FROM transactions AS t
                              LEFT JOIN currency_details AS cd ON cd.ccy = t.currency
                              LEFT JOIN countries AS c ON UPPER(c.code3)= UPPER(t.merchant_country)
                              LEFT JOIN fraudsters as f ON f.user_id = t.user_id
                              """, engine)

    df_transactions.loc[:, 'created_date'] = df_transactions['created_date'].astype('datetime64[ns]')

    df_transactions.loc[:, 'amount_usd'] = df_transactions['amount_usd'] * np.power(10.0, -2)

    return df_transactions

def get_users(engine: sqlalchemy.engine.base.Engine, df_transactions=None):
    if type(df_transactions) == None:
        df_transactions = get_transactions(engine)

    df_users = pd.read_sql("""SELECT u.*, c.name as country_name 
                          FROM users AS u
                          INNER JOIN countries AS c ON c.code = u.country
                          """, engine).set_index('id', drop=False)

    transaction_count_by_user = df_transactions.groupby('user_id')['amount'].count().rename('transaction_count')

    mean_transaction_period = df_transactions.groupby('user_id')['created_date'].apply(lambda x: x.diff().abs().mean())
    mean_transaction_period = mean_transaction_period.astype('timedelta64[h]').rename('mean_transaction_peried')

    df_users = pd.concat([df_users, transaction_count_by_user, mean_transaction_period], axis=1)

    ## Combine transactions and users table by using summary statistics (sum and median)

    usd_total_amount_by_user = df_transactions.groupby('user_id')['amount_usd'].sum()\
                                    .rename('total_amount_usd').astype(np.float32)
    usd_total_amount_by_user = usd_total_amount_by_user.loc[usd_total_amount_by_user.index.intersection(df_users.index)]

    usd_median_amount_by_user = df_transactions.groupby('user_id')['amount_usd'].median()\
                                    .rename('median_amount_usd').astype(np.float32)
    usd_median_amount_by_user = usd_median_amount_by_user.loc[usd_median_amount_by_user.index.intersection(df_users.index)]

    df_users = pd.concat([df_users, usd_total_amount_by_user, usd_median_amount_by_user], axis=1)

    return df_users

def load_features(engine: sqlalchemy.engine.base.Engine, processed_data, processed_labels):
    processed_data.to_sql('features', con=engine, index_label='user_id', if_exists='replace', method=psql_insert_copy)

    processed_labels = pd.DataFrame({'is_fraudster': processed_labels}, index=processed_labels.index)

    processed_labels.to_sql('labels', con=engine, index_label='user_id', if_exists='replace', method=psql_insert_copy)

    with engine.connect() as con:
        con.execute('ALTER TABLE features ADD PRIMARY KEY (user_id);')
        con.execute('ALTER TABLE labels ADD PRIMARY KEY (user_id);')

def get_features(engine, user_id=None):
    """
    Load features from database
    :param user_id: user_id to retrieve, if None retrieve features of all users.
    :return:
    """
    where_clause = ""
    params = None
    if user_id != None:
        where_clause = "WHERE f.user_id=  %(user_id)s"
        params = {'user_id': user_id}

    df_features = pd.read_sql("""SELECT f.*, l.is_fraudster FROM features AS f 
                                 INNER JOIN labels AS l ON l.user_id=f.user_id
                              """ + where_clause, engine,
                              params=params).set_index('user_id', drop=True)
    df_features = df_features.replace({None: np.nan})

    processed_dataset = df_features.drop('is_fraudster', axis=1)
    processed_labels = df_features['is_fraudster']

    return processed_dataset, processed_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_config_path", help="Database config file (.yaml)",
                        default='../misc/database_config.yaml', required=False)
    parser.add_argument("--schema_path", help="Schema file (.yaml)", default='../misc/schemas.yaml', required=False)
    parser.add_argument("--data_path", help="Data path containing csv files of the data that will be loaded into database"
                       , default='../data/', required=False)

    args = parser.parse_args()
    database_config = load_config(args.database_config_path)[0]
    etl(database_config, args.schema_path, args.data_path)