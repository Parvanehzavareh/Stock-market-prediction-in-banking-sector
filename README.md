# Stock Price Prediction Using Machine Learning and Deep Learning
# Project Overview
This repository contains the code and Dataset for a class project titled "Application of Economic Indicators and Historical Data on US Bank Stocks: An ML and DL Forecasting Approach." The project was conducted as part of the MB120 - Applied and Big Data Analytics course at Kristiania University College during Spring 2024.
The project focuses on predicting the stock prices of three major US banks—JPMorgan Chase, Bank of America, and Wells Fargo—using a combination of machine learning (ML) and deep learning (DL) models. The models were trained on a dataset comprising various economic indicators and historical stock prices collected over a twelve-year period from January 1, 2010, to January 1, 2022.
# Project Description
The project aims to identify the most accurate ML and DL models for predicting stock prices by analyzing the impact of key economic indicators. This analysis serves as a foundation for creating actionable buy and sell signals, which are essential for effective risk management and investment decision-making.
The study includes models such as Ridge Regression, Lasso Regression, Decision Tree Regression, Random Forest, XGBoost, LSTM, GRU, and CNN. The performance of these models was evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
# Required Libraries
•	numpy
•	pandas
•	matplotlib
•	seaborn
•	scikit-learn
•	tensorflow
•	keras
•	xgboost
•	yfinance
# Usage
1.	Data Preprocessing: Preprocess the dataset by executing the data_preprocessing.py script, which includes data cleaning, normalization, and feature engineering steps.
2.	Model Training: Run the model_training.py script to train the selected ML and DL models.
3.	Evaluation and Analysis: Execute the evaluation.py script to evaluate model performance using predefined metrics.
4.	Trading Signals and Profit Calculation: Use the trading_signals.py script to generate buy and sell signals and calculate the profit/loss from these signals.
# Models and Methodology
The project employs the following models for stock price prediction:
# Machine Learning Models
•	Ridge Regression
•	Lasso Regression
•	Random Forest
•	Decision Tree Regression
•	XGBoost
# Deep Learning Models
•	Long Short-Term Memory (LSTM)
•	Gated Recurrent Unit (GRU)
•	Convolutional Neural Network (CNN)
•	Sequence-to-Sequence (Seq2Seq)
These models were trained on both technical and economic features, such as moving averages, inflation rate, GDP, and unemployment rate.
# Evaluation
The models were evaluated using the following metrics:
•	Mean Squared Error (MSE)
•	Root Mean Squared Error (RMSE)
•	R-squared (R²)
The evaluation revealed that the Ridge Regression model achieved the highest predictive accuracy with the lowest RMSE, while LSTM and GRU performed better under volatile market conditions.
# Results
Based on the analysis, the Ridge Regression model demonstrated the highest profitability when used to generate buy and sell signals. 
# References
The detailed methodology and analysis can be found in the project report, available upon request. The project utilized various external resources, which are cited in the References section of the report.
# License
This project is for academic purposes only. Please contact for further inquiries.

