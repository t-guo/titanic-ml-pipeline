# Machine Learning Pipeline for Titanic Survival Prediction
A simple machine learning pipeline for Kaggle's [Titanic problem](https://www.kaggle.com/c/titanic).

## Requirement
* pandas==0.23.4
* numpy==1.15.1
* luigi==2.7.8
* scikit_learn==0.20.0
* PyYAML==3.13
* xgboost==0.80

## Execution
Run ```pipeline\main.py``` to execute pipeline. 

Put ```train.csv``` and ```test.csv``` goes under ```data\```. Edit ```config.yaml``` for different experiments.

## Results
~86% CV accuracy.

Best public test score: 81%.


