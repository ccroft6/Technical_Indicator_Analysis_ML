import os
import requests
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset
import hvplot.pandas
import streamlit as st
import numpy as np
from finta import TA # for more info: https://github.com/peerchemist/finta
import yfinance as yf 

import time
import itertools
import random
from functools import reduce
import datetime
from collections import Counter

# local imports
import finta_helper

# scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler

# models
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# The wide option will take up the entire screen.
st.set_page_config(page_title="Technical Analysis Machine Learning",layout="wide")
# this is change the page so that it will take a max with of 1200px, instead
# of the whole screen.
st.markdown(
        f"""<style>.main .block-container{{ max-width: 1200px }} </style> """,
        unsafe_allow_html=True,
)
finta_cache = {}

finta_working_funcs = finta_helper.getWorkingFunctions()
finta_funcs_to_names_map = finta_helper.getFuncsToNamesMap()

if 'last_runs_fa_funcs' not in st.session_state:
    # This is initializing the state.
    st.session_state['last_runs_fa_funcs'] = None

def getYahooStockData(ticker, start_date, end_date):
    """
    Gets data from yahoo stock api. 
    Args:
        ticker: stock ticker
        years: number of years of data to pull
    Return:
        Dataframe of stock data
    """
    result_df = yf.download(ticker, start=start_date,  end=end_date,  progress=False )

    # renaming cols to be compliant with finta
    result_df = result_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    # dropping un-used col
    result_df = result_df.drop(columns=["Adj Close"])
    return result_df

def prepDf(df):
    """
    Does some basic prep work on the dataframe
    Args:
        df: dataframe to prep
    Return:
        dataframe which has been prepped
    """
    
    df["Actual Returns"] = df["close"].pct_change()
    df = df.dropna()
    # Initialize the new Signal column
    #df['Signal'] = 0.0
    df.loc[:,'Signal'] = 0.0
    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    df.loc[(df['Actual Returns'] >= 0), 'Signal'] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    df.loc[(df['Actual Returns'] < 0), 'Signal'] = -1
    return df

def makeSignalsDf(ohlcv_df):
    """
    makes the signal df

    Args:
        ohlcv_df: basic ohlcv styled df
    Return:
        dataframe that is date indexed
    """
    signals_df = ohlcv_df.copy()
    signals_df = signals_df.drop(columns=["open", "high", "low", "close", "volume"])
    return signals_df

def executeFintaFunctions(df, ohlcv_df, ta_functions, start_date,end_date):
    """
    Executes finta functions on a df which is passed in.
    finta reference: https://github.com/peerchemist/finta
    
    Note - so it seems like it's generating these on the fly, which means there's a 
    lot of calculations. Some of these, like DYMI take like 6 seconds to calculate.
    This utilizes a cache variable which is really important in terms of speeding this
    up.

    Args:
        df: a signals df put all of the new cols on
        ohlcv_df: the standard ohlov df
        ta_fuctions: a list of finta functions to call,
        start_date: start date of the query for the sake of caching.
        end_date: end date of the query for the sake of caching.
    Return:
        dataframe with newly appended finta data.

    """
    
    for ta_function in ta_functions:
        # dynamically calling the TA function.
        _cache_key = f"{ta_function}-{start_date}-{end_date}"
        
        try:
            if _cache_key in finta_cache:
                # some of these functions are expensive to re-generate. trying to do a
                # cache here to avoid doing the same expensive calculations again and again.
                ta_result = finta_cache[_cache_key]
            else:
                # calling the actual finta function:
                # at this point, we have the string version of the finta function name that we want
                # to call. The getattr() function is a way to get a reference to function on an
                # module if just have the string representation of name. 
                #
                # for example, if we are trying to call the TA.sma() function, at this point we
                # will have the 'ta_function' variable with the value of 'sma'. given that, if we call
                # `finata_func = getattr(TA, ta_function)`, it will return a reference to TA.sma()
                # without actually calling it, and then store it as the finta_func variable. From
                # there, we can then excute it on the following line with the ohlcv_df that is
                # necessary to call it.
                finta_func = getattr(TA, ta_function)
                ta_result = finta_func(ohlcv_df)

                finta_cache[_cache_key] = ta_result

            # finta functions results vary in terms of data type. Sometimes, it will return
            # a single column of data stored in a panada series. Other times, like with Bollinger
            # bands, it will return three seperate columns of data inside a panda dataframe.
            # this next bit detects what finta returned, and then adds the columns accordingly.
            if isinstance(ta_result, pd.Series):
                df[ta_function] = ta_result
            elif isinstance(ta_result, pd.DataFrame):
                for col in ta_result.columns:
                    df[col] = ta_result[col]
        except Exception as e:
            st.write("Error - failed to execute: ", ta_function)
            st.write("Error - actual error: ", e)
    df.dropna(inplace=True)
    
    indicators=list(df.columns)
    indicators.remove("Actual Returns")
    indicators.remove("Signal")

    return (df, indicators)

def createScaledTestTrainData(df, indicators, scaler_name):
    """
    created scaled training and test data.

    Args:
        df: data frame
        indicators: all of the indicator data to scale
    Return:
        tuple(X_train_scaled, X_test_scaled, y_train, y_test)
    """
    X = df[indicators].shift().dropna()
    y = df['Signal']
    y=y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle=False)

    # creating the actual scaler based on the scaler_name that is passed in.
    scaler = None
    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif scaler_name == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaler_name == 'PowerTransformer':
        scaler = PowerTransformer()
    elif scaler_name == 'QuantileTransformer':
        scaler = QuantileTransformer(output_distribution="normal")
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def executeSVMModel(X_train_scaled, X_test_scaled, y_train, y_test, signals_df ):
    """
    executs the svm model on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, testing_report)
    """
    
    model = svm.SVC()
    model = model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    testing_report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, testing_report

def executeRandomForest(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    """
    executs the random forest on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, testing_report)
    """
    rf_model = RandomForestClassifier(random_state=10)
    rf_model = rf_model.fit(X_train_scaled, y_train)
    pred = rf_model.predict(X_test_scaled)
    testing_report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, testing_report

def executeAdaBoostClassifier(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    """
    executs the AdaBoost Classifier on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, testing_report)
    """
    #ad_model = AdaBoostClassifier(n_estimators=100)
    ad_model = AdaBoostClassifier()
    ad_model = ad_model.fit(X_train_scaled, y_train)
    pred = ad_model.predict(X_test_scaled)
    testing_report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)

    # Add the AdaBoost model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, testing_report

def executeNaiveBayes(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    """
    executs the naive bayes on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, report)
    """
    
    model = GaussianNB()
    model = model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, report


def execute(ticker, scaler, start_date, end_date, indicators_to_use=[], rerun=False):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.
    Args:
        ticker: ticker to use
        indicators_to_use: indicators to use
        years: # of years of data to base all of this on.
    Return:
        None
  
    """

    # define percentage calculation for columns 
    def percent_column(model, actual):
        a = (model - actual) * 100
        results = ("{:.0f}%".format(a))
        return results

    # Getting the stock data
    ohlcv_df = getYahooStockData(ticker.upper(), start_date, end_date)

    #prepping the stock data
    ohlcv_df = prepDf(ohlcv_df)

    ta_functions = random.choices(finta_working_funcs, k=5)
    if indicators_to_use:
        ta_functions = indicators_to_use
    elif rerun == True:
        ta_functions = st.session_state['last_runs_fa_funcs']
        
    st.session_state['last_runs_fa_funcs'] = ta_functions

    names = [finta_funcs_to_names_map[n] for n in ta_functions]

    #this is generating all of the combinations of the ta_functions. 
    ta_func_combinations = []
    for k in range(len(ta_functions)):
        ta_func_combinations.extend(itertools.combinations(ta_functions, k+1))

    st.write(f"Testing {len(ta_func_combinations)} different combinations of these indicators: ", ", ".join(names))
    
    # this is prepping the final results df    
    top_ten_results_df = pd.DataFrame(columns=["Variation", "SVM Returns", f"SVM vs. {ticker}", "Random Forest Returns", f"RF vs. {ticker}", "AdaBoost Returns", f"AD vs. {ticker}", "Naive Bayes Returns", f"NB vs. {ticker}" ])

    # all of the results dfs should be stored in this map for future reference
    all_combinations_result_map = {}


    classification_report_result_map = {}

    # this is really important. some of the finta functions 
    # take a long time. having a cache really speeds it up
    finta_cache = {}

    actual_returns_for_period = None
    
    for ta_func_combination in ta_func_combinations:

        perm_key = ",".join(ta_func_combination)

        # it's lame to do this every time, but I've experienced so many reference errors with not
        # trying to re-instiate this every loop. Yes, it's lame, but this is a bit more garenteed
        # to work.
        signals_df = makeSignalsDf(ohlcv_df)
        finta_signals_df, indicators = executeFintaFunctions(signals_df, ohlcv_df, ta_func_combination, start_date, end_date)

        X_train_scaled, X_test_scaled, y_train, y_test = createScaledTestTrainData(finta_signals_df, indicators, scaler)

        svm_predictions_df, svm_testing_report = executeSVMModel(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
        rf_predictions_df, rf_testing_report = executeRandomForest(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
        ad_predictions_df, ad_testing_report = executeAdaBoostClassifier(X_train_scaled, X_test_scaled, y_train, y_test,signals_df)
        nb_predictions_df, nb_testing_report = executeNaiveBayes(X_train_scaled, X_test_scaled, y_train, y_test,signals_df)

        svm_final_df = (1 + svm_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()
        rf_final_df = (1 + rf_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()    
        ad_final_df = (1 + ad_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()    
        nb_final_df = (1 + nb_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()
        
        # at this point we have all of our results. This next bit is a way to rename the different cols
        # and then merge them into a single dataframe which we can use to chart later in the results.
        rf_final_df.drop(columns=['Actual Returns'], inplace=True)
        ad_final_df.drop(columns=['Actual Returns'], inplace=True)
        nb_final_df.drop(columns=['Actual Returns'], inplace=True)

        svm_final_return = svm_final_df.iloc[-1]["Strategy Returns"]
        rf_final_return = rf_final_df.iloc[-1]["Strategy Returns"]
        ad_final_return = ad_final_df.iloc[-1]["Strategy Returns"]
        nb_final_return = nb_final_df.iloc[-1]["Strategy Returns"]

        svm_final_df.rename(columns={'Strategy Returns': 'SVM Returns'}, inplace=True)
        rf_final_df.rename(columns={'Strategy Returns': 'Random Forest Returns'}, inplace=True)
        ad_final_df.rename(columns={'Strategy Returns': 'AdaBoost Returns'}, inplace=True)
        nb_final_df.rename(columns={'Strategy Returns': 'Naive Bayes Returns'}, inplace=True)        

        dfs_to_merge = [svm_final_df, rf_final_df,  ad_final_df, nb_final_df]
        merged_df = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs_to_merge)
       
        _key = ",".join([finta_funcs_to_names_map[n] for n in ta_func_combination])

        # fixme - it's lame to have to reset this every iteration
        actual_returns_for_period = svm_final_df.iloc[-1]["Actual Returns"]

        # now that all of the results are in a single dataframe, we're storing the merged_df in a map so that
        # it could possibly be referenced later to display the chart later.
        all_combinations_result_map[_key] = merged_df        
        
        classification_report_result_map[_key] = [svm_testing_report, rf_testing_report, nb_testing_report, ad_testing_report]

        # the next 3 lines is a way to manually add a row to a dataframe
        top_ten_results_df.loc[-1] = [_key,
        svm_final_return, percent_column(svm_final_return, actual_returns_for_period), 
        rf_final_return, percent_column(rf_final_return, actual_returns_for_period),  
        ad_final_return, percent_column(ad_final_return, actual_returns_for_period),
        nb_final_return, percent_column(nb_final_return, actual_returns_for_period)]

        top_ten_results_df.index = top_ten_results_df.index + 1
        top_ten_results_df = top_ten_results_df.sort_index()

       
    top_ten_results_df = top_ten_results_df.sort_values(by=["SVM Returns", "Random Forest Returns", "AdaBoost Returns", "Naive Bayes Returns"], ascending=False)

    st.write(f"Return for {ticker} over the testing period is {round(actual_returns_for_period,4)}")
    st.write("Top 10 Models:")

    hide_table_row_index = """<style>tbody th {display:none} .blank {display:none} </style> """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)    

    st.table(top_ten_results_df.head(10))
    
    for perm_key in top_ten_results_df["Variation"][:10]:
        st.write(f"Results for: {perm_key}")
        st.line_chart(all_combinations_result_map[perm_key])
        # Display classification report
        with st.expander("Classification Report Comparison"):
            st.write('SVM Report')
            st.table(pd.DataFrame(classification_report_result_map[perm_key][0]))
            st.write('RandomForest Report')
            st.table(pd.DataFrame(classification_report_result_map[perm_key][1]))
            st.write('AdaBoost Report')
            st.table(pd.DataFrame(classification_report_result_map[perm_key][2]))
            st.write('Naive Bayes Report')
            st.table(pd.DataFrame(classification_report_result_map[perm_key][3]))


def main():
    """
    Main function of this app. Sets up the side bar and then exectues the rest of the code.

    Returns:
        None
    """
   
    st.title("Technical Indicator Analysis with ML")

    #st.sidebar.info( "Select the criteria to run:")

    # reversing this again
    valid_indicators = {v: k for k, v in finta_funcs_to_names_map.items()}

    valid_indicator_names = valid_indicators.keys()

    all_scalers = ['StandardScaler','MinMaxScaler','MaxAbsScaler',
                   'QuantileTransformer','PowerTransformer', 'RobustScaler']
    
    selected_stock = st.sidebar.text_input("Choose a stock:", value="SPY")
    selected_scaler = st.sidebar.selectbox("Choose a Scaler:", all_scalers)

    today = pd.to_datetime('today').normalize()
    
    start_date = st.sidebar.date_input("Start Date", value=(today - DateOffset(years=10)), max_value=today)
    end_date = st.sidebar.date_input("End Date", max_value=today)    
    
    st.sidebar.markdown("---")    
    named_selected_indicators = st.sidebar.multiselect("TA Indicators to use:", valid_indicator_names)

    selected_indicators = []

    for named_indicator in named_selected_indicators:
        selected_indicators.append(valid_indicators[named_indicator])

    if st.sidebar.button("Run"):
        with st.spinner('Executing...'):
            execute(selected_stock, selected_scaler, start_date, end_date, selected_indicators)
    st.sidebar.markdown("---")
    st.sidebar.write("This will randomly choose 5 indicators")
    if st.sidebar.button("I'm feeling lucky"):
        with st.spinner('Executing...'):
            execute(selected_stock, selected_scaler, start_date, end_date)
    if st.sidebar.button("Re-run last"):
        with st.spinner('Executing...'):
            execute(selected_stock, selected_scaler, start_date, end_date, [], True)
        
main()
