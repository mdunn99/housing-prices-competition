# Housing Pricing Competition: Random Forest Regressor
This program was written as a contender in the K\*ggle (censored to prevent cheating users) ["Housing Prices Competition."](https://www.kaggle.com/competitions/home-data-for-ml-course/) In the associated course, I learned about random forest regressors and feature selection. I used this knowledge and leveraged the sklearn libraries to systematically identify an ideal number of leaf nodes, select ideal features to pass through the regression model, and do this with sklearn's metrics and train_test_split modules.

`main.py` uses pandas to select initial X and Y features and convert categorical labels into integer representations, effectively allowing the regressor to identify meaningful trends with categorical data. The program creates a model from the `train.csv` dataset and then tries its hand at new data provided by K\*ggle in `test.csv`. It then outputs a predictions dataset, `test_predictions.csv` for easy reading by the competition's grading system, using K\*ggle's requested index.

# Use
1. Clone this repository and select it
2. Install the necessary dependencies in `requirements.txt`
3. Run `main.py`