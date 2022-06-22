![Boom-Bull-Stock-Exchange-Bear-World-Economy-913982](https://user-images.githubusercontent.com/95719899/164332595-9cf0e024-aab1-4d3b-a11c-8c47ee122b9e.jpg)

---

# Machine_Learning_Algorithmic_Technical_Indicator_Analysis

**LAUNCH APP**: https://share.streamlit.io/lariannrupp/machine_learning_algorithmic_trading_bot/main/streamlit.py

---

## Purpose: 

The purpose of the Project is to create a streamlit dashboard; which given a specific stock, performs a comparative analysis of machine-learning model design in a testing environment. It is called the "Technical Indicator Analysis with ML".

The dashboard demonstrates the framework for a more powerful terminal that can take a stock and design a machine learning model for it. 


---

## Installation Guide

The app is deployed to a Streamlit interface, so to use the app, no installation is required. Simply launch the link at the top of the README.

To run it from local computer, clone repo. From the Terminal, activate conda envirnoment; change directory to repo folder,
and type `streamlit run streamlit.py`. This will launch a web browser with a local URL such as http://localhost:8501. *Note that you will need to install some libraries/dependencies if you do not already have them. You can do so by typing `pip install requirements.txt` into the terminal.* 

---

## Data

OHLVC (open, high, low, volume, close) stock data is pulled from the Yahoo Finance API. You can choose a start date and an end date to analyze.

The data undergoes a standard train, fit, predit process with the training window being 60% of the data, and the testing window being 40% of the data. 

By default, the app scales X features with StandardScaler(), but within the app, the user can test out different data scaling methods. 

---

## Selected Machine Learning Models

The following machine learning models were selected because they are common, supervised models with binary classification systems:

- SVM (Support Vector Machine)

- Random Forest

- Naive Bayes

- AdaBoost 

---

## Approach

The User can test combinations of up to 54 technical indicators. However, testing all 54 indicators at once would take a long time and a personal computer may not have the power for this request. We recommend using 5 indicators, which results in 31 possible combinations (i.e., each indicator by itself, all 5 indicators together, and all combinations of 2, 3, and 4 indicators). Then, the top 10 best-performing combinations are displayed. 

For users who would like to explore combinations of a random 5 indicators, they can use the **I'm feeling lucky** button. 

The **Re-run last** button allows the user to select a different scaler, for example, and make comparisons by testing the same indicators from the last run.

Dashboard: Select parameters to perform Technical Indicator Analysis with ML.

<img width="170" alt="Dashboard" src="https://user-images.githubusercontent.com/93550651/164949541-0af2877d-d77c-4679-b9f8-e771067527d4.png">

Results appear as follows:

<img width="716" alt="Dashboard Results" src="https://user-images.githubusercontent.com/93550651/164949764-575de6ee-9724-456e-a55f-8f10ec259a68.png">


---

## Performance Evaluation and Backtesting

Cumulative return plots compare model performance to market performance. Additionally, the table for "Top 10 Models" compares cumulative returns as both a ratio and a percentage. 


![Screenshot 2022-04-20 171355](https://user-images.githubusercontent.com/95719899/164341560-ee00d663-34b1-4df4-81a2-f6c466ac306f.jpg)



To backtest the models and compare trade accuracies, the user can drop down the **Classification Report Comparison** button. Please note that trade accuracies can oftentimes be a better metric of model performance than cumulative returns. 

![Screenshot 2022-04-20 171609](https://user-images.githubusercontent.com/95719899/164341569-c4fedbb2-2749-48a9-b6b7-c4e8433a484f.jpg)

---

## Contributors:

### Development Team
Leigh Badua

John Batarse

Catherine Croft

Jing Pu

Jason Rossi

Lari Rupp


### Collaborators

University of California Berkelely Fintech Bootcamp (Project 2, 2022)

Trilogy Education LLC.

Kevin Lee (Instructor)

Vincent Lin (TA)

---


## Technologies

A Python 3.9.7 (ipykernal) was used to write this app.

![python-logo-master-v3-TM-flattened](https://user-images.githubusercontent.com/95719899/164334658-d32c6762-b35d-4ae3-8d87-f054388941e7.png)
![Pandas_logo svg](https://user-images.githubusercontent.com/95719899/164334292-8243632d-1274-4c4f-ba36-cbf71dc14309.png)
![YahooFinance](https://user-images.githubusercontent.com/95719899/164334383-5f613f77-fb14-4b8c-80a7-882241baf76a.png)
![1200px-Finta_Logo](https://user-images.githubusercontent.com/95719899/164334464-705a5167-9385-4f93-91b4-5afc74a0ea24.png)
![1200px-Scikit_learn_logo_small svg](https://user-images.githubusercontent.com/95719899/164334470-dac38a18-1d42-4bfe-abfe-7f681677a8ff.png)
![streamlit_logo](https://user-images.githubusercontent.com/95719899/164334479-b14755bc-7525-4f9b-aeaf-6e56df94f49d.png)


---

## License

Creative Commons Zero

This is a truly open-source project under the Creative Commons Zero license which is free for use for everyone.

We ask that you please credit the team with the following IEEE citation:

> L. Badua, J. Batarse, C. Croft, J. Pu, J. Rossi, L. Rupp, “Machine_Learning_Algorithmic_Trading_Bot,” University of California Berkeley Extension Fintech Bootcamp, Berkeley, California, USA, Fintech Bootcamp Project 2, 2022. https://github.com/lariannrupp/Machine_Learning_Algorithmic_Trading_Bot (accessed month day, year).
