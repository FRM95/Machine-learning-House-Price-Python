# Python Machine Learning House price prediction

Machine Learning aplication created for House Price Kaggle competition as part of my final Big Data Master degree project. 

Model prediction with Gradient Boosting and Grid Search for Hyperparameter tuning in HousePriceModel.py.

Implemented Kaggle API functionality for Python at kaggleMethods.py.

Functionality:

1. Join Competition

Join House Price Competition as a team: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/tutorials

2. Get Kaggle API credentials 

Create New API Token in your Kaggle Account Settings and download kaggle.JSON file. 
Open it, copy username and key values and place them into KaggleMethods.py autUser dictionary to set them as Kaggle enviroment variables.

Example: autUSer = {"username": '<JSON username>' , "key": '<JSON key>'}

Alternative: You can add this JSON file into your '/Users/User/.kaggle' directory after installing Kaggle library and comment autUser and os.environ lines in kaggleMethods.py file.

3. Install requirements

Install modules with: 'pip install -r requirements.txt'

4. Execute App

Execute via command line defineModel.py to create Gradient Boosting Model with test and train data of House Price Kaggle competition.

Example: python defineModel.py example_submission.csv

Alternative: Before executing defineModel.py you can download data competition files manually and place them into Kaggle_data directory.

5. Submit prediction file and get leaderboard public score.

Execute via command line makeSubmission.py to submit your predictions generated at Submissions/example_submission.csv.

Example: python makeSubmission.py example_submission.csv. 
