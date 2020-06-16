import numpy as np
import shutil
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


def productIDs():
    """
    dummy product replacement for real product for privacy purposes. 
    """
    return ['prod1','prod2']

def price_product_dict():
    fname = '../../original_data/productprices.csv'
    df = pd.read_csv(fname)
    price_dict = {df['product'][i]:df['prices'][i] 
                for i in range(len(df['prices']))}
    return price_dict


def get_features(df):
    X = df[['delta_price','buybox','sessions','pageviews']]
    y = df['demand']
    return df['weeks'],X,y

def update_sales(datafile_path):
    # copy old file and save it for backup
    current_sales_fname = datafile_path + '/sales.csv'
    backup_sale_fname = datafile_path + '/sales_backup.csv'
    shutil.copy(current_sales_fname,backup_sale_fname)
    df_sales = pd.read_csv(current_sales_fname)

    new_sales_fname = datafile_path + '/sales_thisweek.csv'
    df_new = pd.read_csv(new_sales_fname)
    df_new['weeks'] = np.max(df_sales['weeks']) + 1

    # update sales 
    df = pd.concat([df_sales,df_new])
    
    # Save most up to date sales data
    df.to_csv(current_sales_fname)
    return df

def sql_select_data(prod_id):
    dbname,username,pswd = read_credential()

    engine = create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))

    # connect to database. 
    con = None
    con = psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)

    # query:
    sql_query = """
    SELECT * FROM sales_data_table WHERE product_id='%s';
    """ % prod_id


    sales_data = pd.read_sql_query(sql_query,con)
    return sales_data


def read_credential():
    fname = './sql_credential.dat'
    credential = []
    with open(fname) as f:
        for line in f:
            line = line.rstrip().split(' ')
            credential.append(line[1])
    return credential