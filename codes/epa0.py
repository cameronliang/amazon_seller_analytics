import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




fname = '../original_data/Revised By ASIN-Datail Page Sales and Traffic by Parent Item.csv'
df = pd.read_csv(fname)

df['Page Views'] = df['Page Views'].str.replace(',','').astype(float)
# this works now. 

for view in df['Page Views']:
    print(view)