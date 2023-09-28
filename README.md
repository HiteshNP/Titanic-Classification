# Titanic-Survival-Prediction

<img src="https://wallpapercave.com/wp/wp2563741.jpg](https://raw.githubusercontent.com/Masterx-AI/Project_Titanic_Survival_Prediction_/main/titanic.jpg">

Hello Everyone,

Here is My Classification Project based on Predicting Survival of Passengers.

## Dataset

I used Titanic Dataset avaliable on Kaggle.

**Link to the Dataset :** [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset?select=Titanic-Dataset.csv)

## Problem Statement

- The sinking of the Titanic is one of the most infamous shipwrecks in history.

- On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.

- Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

- While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

- In this challenge, we have to build a predictive model that answers the question : “what sorts of people were more likely to survive?” using Passenger Data.

## Table of Contents

- [Setting up the Enviroment](#setting-up-the-enviroment)
- [Libraries required for the Project](#libraries-required-for-the-project)
- [Getting started with Repository](#getting-started)
- [Steps involved in the Project](#steps-involved-in-the-project)
- [Conclusion](#conclusion)
- [Link to the Notebook](#link-to-the-notebook)

## Setting up the Enviroment

Jupyter Notebook is required for this project and you can install and set it up in the terminal.

- Install the jupyter notebook - `pip install jupyter notebook`

- Run the Notebook - `jupyter notebook`

## Libraries required for the Project

**NumPy**

- Go to Terminal and run this code - `pip install numpy`

- Go to Jupyter Notebook and run this code from a cell - `!pip install numpy`

**Pandas**

- Go to Terminal and run this code - `pip install pandas`

- Go to Jupyter Notebook and run this code from a cell - `!pip install pandas`

**Matplotlib**

- Go to Terminal and run this code - `pip install matplotlib`

- Go to Jupyter Notebook and run this code from a cell - `!pip install matplotlib`

**Sklearn**

- Go to Terminal and run this code - `pip install sklearn`

- Go to Jupyter Notebook and run this code from a cell - `!pip install sklearn`

## Getting Started

- Clone the repository to your local machine using the following command :
```
git clone https://github.com/HiteshNP/Titanic-Classification.git
```

## Steps involved in the Project

**Data Cleaning**

- Removing Null Values in the Age Columns and replacing them with Mean Age by using fillna().mean().

- Dropping Cabin Columns as it contains Many Null Values.

- Dropping Text Columns from our Dataset because our Model only works on Numerical Data.

- Creating Dummies Value for Sex Column and Converting it into a DataFrame and Concatenating it with Orignal DataFrame.

**Model Building**

- Firstly I have defined Dependent and Independent Variables for our Traning and Testing.

- I have splitted data into Traning and Testing Set by using Train Test Split.

- Then I Trained the Model with X_train and y_train and checked the Score.

- And Finally I predicted the Result from my Trained Model.

## Conclusion

- In conclusion, the Titanic Survival Prediction Project was an exciting endeavor where I applied Logistic Regression, Support Vector Machines, Naive Bayes, KNN and Decision Tree to predict the survival of passengers aboard the Titanic.

- The Naive Bayes model achieved Accuracy Score:0.7686, Logistic Regression model achieved Accuracy Score:0.7611,	Decision Tree model achieved Accuracy Score: 0.7425, Support Vector Machines model achieved Accuracy Score:0.6604, KNN model achieved Accuracy Score:0.6604, indicating a reasonably good level of accuracy in predicting survival outcomes.

- Therefore, by employing a Naive Bayes model, one can attain the highest level of accuracy when predicting survival outcomes.

## Link to the Notebook:
[Titanic Survival Prediction](https://github.com/HiteshNP/Titanic-Classification/blob/master/Titanic%20Predication.ipynb)
