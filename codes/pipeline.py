import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def AddData(datafile_path):
    # copy old file and save it for backup
    current_sales_fname = datafile_path + '/sales.csv'
    backup_sale_fname = datafile_path + '/sales_backup.csv'
    backup_sales = shutil.copy(current_sales_fname,backup_sale_fname)
    df_sales = pd.read_csv(current_sales_fname)

    new_sales_fname = datafile_path + '/sales_thisweek.csv'
    df_new = pd.read_csv(new_sales_fname)
    df_new['weeks'] = np.max(df_sales['weeks']) + 1

    # update sales 
    df = pd.concat([df_sales,df_new])
    
    # Save most up to date sales data
    output_fname = datafile_path + '/sales_new.csv'
    df.to_csv(output_fname)
    return df





if __name__ == '__main__': 


    # input -- current filename 
    # this includes all the features, and demand. 
    # then fit the model 
    # then make prediction of revenue, given the price. 
    # compute 
    #current_sales_fname = '../../cleaned_data/simulations/sales.csv'
    datafile_path = '../../cleaned_data/simulations/'
    df = AddData(datafile_path)