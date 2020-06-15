################################################################################
#
# SimulationModel.py 	(c) Cameron Liang 
#						Insight Data Science Fellowship Program 
#     				    cameron.liang@gmail.com
#
# Produce mock sales data and perform model fit to test convergence of 
# best-fit parameters in model and validity of online-learning methodology 
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def _mock_params(product):
    """
    parameters of the mock model 
    a_feature: demand = a_feature * feature; slope of the line (linear model)

    model_demand = a_constant + params * features (inner product)
    """
    a_constant  = 50  # combination of constants from the features in linear model 
    a_price = -20.5     # elasticity of product given price. 
    a_buybox_perc = 0.2 # how buyers buy given buybox % 
    a_session = 0.05  # conversion rate of session into units order 
    a_pageviews = 0.03 # conversion rate of pageviews into units order
    params_prod1 = np.array([a_constant,a_price,a_buybox_perc, a_session,a_pageviews])

    a_constant  = 20  # combination of constants from the features in linear model 
    a_price = -10.5     # elasticity of product given price. 
    a_buybox_perc = 0.3 # how buyers buy given buybox % 
    a_session = 0.015  # conversion rate of session into units order 
    a_pageviews = 0.06 # conversion rate of pageviews into units order
    params_prod2 = np.array([a_constant,a_price,a_buybox_perc, a_session,a_pageviews])
    
    params_dict = {'prod1':params_prod1, 'prod2':params_prod2}

    return params_dict[product]

def generate_model_data(product_id):
    """
    todo: add holiday effects in each generation of data. 
    D_sub = Dt + D(x), where D_sub is modulated by Dt. 

    Underlying model for simulated data

    Parameters:
    ----------
    params: array_like
        coefficients for each feature of the model. 
    x: array_like
        features that the model depends on; the dependent 
        variables. 
        session: number of unique visitor 
        page_views: number of page views combined all visitors 
        buy_box_perc: percetange on the buybux shared by competitors. 
        price: price in dollars
    Returns
    ----------
    demand: float
        number of units ordered
    """
    # user input: 
    percent_price_change = 0.2 # percentage allowed to change. 
    # 20% deviation of price 
    sample_size = 52 # weeks of data, each has feature = [price, buybox, session, page_views]
    price_lower_bound = (-percent_price_change) # or manufacturer allowed lowest price. 
    price_upper_bound = (percent_price_change)

    delta_prices = np.random.normal(0,percent_price_change,sample_size)
    
    buybox = np.random.uniform(0,1,sample_size)*100 # buybox in percentage. 
    
    max_session = 100
    sessions = np.random.uniform(0,max_session,sample_size)

    max_page_views = 100
    page_views = np.random.uniform(0,max_page_views,sample_size)
    
    
    params = _mock_params('prod1')
    
    features = np.array([delta_prices, buybox, sessions, page_views]).T

    # shorten this in pythonic way 
    model_demand = np.zeros(len(features))
    demand_price = np.zeros(len(features))
    demand_buybox = np.zeros(len(features))
    demand_sessions = np.zeros(len(features))
    demand_pageviews = np.zeros(len(features))
    for i,feature in enumerate(features):
        #model_demand[i] = params[0] + np.product((params[1:], feature))
        demand_price[i]     = np.product((params[1], delta_prices[i]))
        demand_buybox[i]    = np.product((params[2], buybox[i]))
        demand_sessions[i]  = np.product((params[3], sessions[i]))
        demand_pageviews[i] = np.product((params[4], page_views[i]))
        
        model_demand[i] = params[0] + demand_price[i] + demand_buybox[i] + demand_sessions[i] + demand_pageviews[i]

    # gaussian noise 10% std of model demand. 
    mean = 0 
    std = 1.0 # in absolute units. i.e., not relative. 
    noise = np.random.normal(mean,std, size = sample_size)

    model_demand += noise 
    weeks = np.arange(1,sample_size+1,1)
    model_data = np.array([weeks,delta_prices,buybox,sessions,page_views,model_demand])
    df = pd.DataFrame(data=model_data.T, columns=['weeks','delta_price','buybox','sessions','pageviews','demand'])
    df['product_id'] = product_id
    output_fname = '../../cleaned_data/simulations/sales.csv'
    if os.path.isfile(output_fname):
        with open(output_fname,'a') as f: 
            df.to_csv(f,index=False, header=False)
    else:
        df.to_csv(output_fname,index=False)
    return 

def FitModel():
    """
    Simulation tests for parameters convergence. 
    """
    from sklearn.linear_model import LinearRegression
    
    # read in data 
    fname = '../../cleaned_data/simulations/sales.csv'
    df = pd.read_csv(fname)
    X = df.drop(['weeks','demand'],axis=1) # remove weeks and demands, select features only
    y = df['demand']

    # Fit model and record parameters 
    n_params = len(X.iloc[0]) + 1
    params = np.zeros((len(y)-2,n_params)) # shape = (198,5) = (n_weeks,n_params)
    model = LinearRegression()
    for i in range(2,len(y)): # 198

        temp_x = X.iloc[:i]
        temp_y  = y.iloc[:i]
        model.fit(temp_x,temp_y)

        param = np.concatenate(([model.intercept_],model.coef_))
        params[i-2] = param

    
    df = pd.DataFrame(data=params,columns=['intercept','price_coef','buybox_coef','sessions_coef','pageviews_coef'])
    df.to_csv('../../cleaned_data/simulations/params.csv')

    return params


if __name__ == '__main__':
    import sys 
    product_id = sys.argv[1]
    generate_model_data(product_id)
    #FitModel()

    