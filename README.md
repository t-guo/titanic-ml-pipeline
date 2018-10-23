# Machine Learning Pipeline for Titanic Survival Prediction
A simple machine learning pipeline for Kaggle's [Titanic problem](https://www.kaggle.com/c/titanic).

## Requirement
* pandas==0.23.4
* numpy==1.15.1
* luigi==2.7.8
* scikit_learn==0.20.0
* scipy==1.1.0
* PyYAML==3.13
* xgboost==0.80

## Pipeline
Configuration based modeling workflow.
#### Tasks:
1. Feature Engineering
    * Data cleaning and imputation
    * Feature extraction
    * Encoding categorical variables 
    * Feature selection with supported [sklearn feature selection estimator](http://scikit-learn.org/stable/modules/feature_selection.html) 
    
    *(NOTE: adding feature selection will reduce cv accuracy, but it reduces overfitting and tends to improve kaggle score on the test set)*
    
2. Modeling
    * Grid Search with cross validation on specified estimators
    * Ensemble best model from each estimator
    * Support models - any estimator from [sklearn](http://scikit-learn.org/stable/index.html) and [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
3. Making predictions
    * Apply pipeline process on test set

## Execution
Run ```pipeline\main.py``` to execute pipeline. 

Put ```train.csv``` and ```test.csv``` goes under ```data\```. Edit ```config.yaml``` for different experiments.

## Results
~83% CV accuracy. 

Best public test score: 82%.


