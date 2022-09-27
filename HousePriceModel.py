# Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time


class HousePriceModel(object):

    __data_total = pd.DataFrame()

    # Constructor class defines train, test and target
    def __init__(self):
        self.df_train = pd.read_csv('./Kaggle_data/train.csv')
        self.df_test = pd.read_csv('./Kaggle_data/test.csv')
        self.target = self.df_train[['Id', 'SalePrice']].copy()
        self.df_train.drop(['SalePrice'], axis=1, inplace=True)

    # Checks if train and test columns are equal
    def checkdata(self):
        if self.df_train.columns.all() != self.df_test.columns.all():
            raise Exception('Test and Train columns are not similar')
        # else we define total dataset
        self.df_total = pd.concat([self.df_train, self.df_test])
        self.df_total.reset_index(drop=True, inplace=True)
        # class variable use example
        self.__class__.__data_total = self.df_total
        print('Check: Test and train data columns are similar\n')

    # Define type columns lists
    def defvariables(self):
        self.numeric_cols = self.df_train.select_dtypes([np.number]).columns.values.tolist()
        self.numeric_cols.remove('Id')
        self.categoric_cols = self.df_train.select_dtypes(include='object').columns.values.tolist()

    # Columns where 'NA' means no value, new value -> 'NA'
    __colsNA = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'GarageFinish', 'GarageCond',
                'GarageType', 'GarageQual', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'BsmtFinType2', 'BsmtFinType1']
    # Columns where 'NA' means 0, new value -> 0
    __cols0 = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
               'BsmtFinSF2', 'TotalBsmtSF']
    # Columns where 'NA' new value -> mode
    __colsMode = ['MSZoning', 'Functional', 'Utilities', 'BsmtHalfBath', 'BsmtFullBath', 'SaleType', 'Exterior1st',
                  'Exterior2nd', 'KitchenQual', 'Electrical']
    # Other columns transformations
    __colsLeft = []

    # Left columns append
    def __leftcols(self):
        for x in self.__class__.__data_total.columns.tolist():
            if x not in self.__class__.__colsNA + self.__class__.__cols0 + self.__class__.__colsMode and x != 'Id':
                self.__class__.__colsLeft.append(x)

    __imput = lambda x, data, val: data[x].fillna(value=val, inplace=True)

    # Making NA imputations
    def imputna(self):
        self.__leftcols()
        # Imputa todas los NA's en df_total según el tipo de columna
        for i in self.__colsNA:
            if i in self.df_total:
                self.__class__.__imput(i, self.df_total, 'NA')
            else:
                pass

        for i in self.__cols0:
            if i in self.df_total:
                self.__class__.__imput(i, self.df_total, 0)
            else:
                pass

        for i in self.__colsMode:
            if i in self.df_total:
                self.__class__.__imput(i, self.df_total, self.df_total[i].mode()[0])
            else:
                pass

        for i in self.__colsLeft:
            if i in self.df_total:
                if self.df_total[i].dtype == 'object':
                    self.__class__.__imput(i, self.df_total, self.df_total[i].mode()[0])
                else:
                    self.__class__.__imput(i, self.df_total, self.df_total[i].mean())
            else:
                pass
        print('1. Applied NA imputation to total data\n')

    # Categoric columns definition
    __categoric_cols_ord = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                            'GarageCond', 'PoolQC']

    # Manual transformation of categoric columns
    def colstonum(self):
        self.df_total = self.df_total.replace({
            'GarageType': {'NA': 0, 'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3, 'Basment': 4, 'CarPort': 5, '2Types': 6},
            'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
            'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'MiscFeature': {'NA': 0, 'TenC': 3, 'Elev': 1, 'Gar2': 2, 'Shed': 4, 'Othr': 5},
            'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'Fence': {'NA': 0, 'GdPrv': 2, 'MnPrv': 1, 'GdWo': 4, 'MnWw': 3},
            'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4},
            'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
            'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},
            'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2},
            'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
            'Street': {'Grvl': 1, 'Pave': 2},
            'CentralAir': {'N': 0, 'Y': 1}
        })
        print('2. Applied manual transformations to categorical columns\n')

    # Categorical columns can be transformed with labelEncoder
    def labencoder(self):
        for i in self.df_total.columns[self.df_total.dtypes == 'object']:
            le = LabelEncoder()
            le.fit(list(self.df_total[i].unique()))
            self.df_total[i] = le.transform(self.df_total[i])
        print('3. Applied Label Encoder\n')

    # OneHotEncoding transformations
    def onehot(self, value: int = 5):
        oneHotCols = []
        for i in self.categoric_cols:
            if i not in self.__categoric_cols_ord and (len(self.df_total[i].value_counts()) <= value):
                oneHotCols.append(i)
        self.df_total = pd.get_dummies(data=self.df_total, columns=oneHotCols, drop_first=True)
        print('4. Applied OneHot to columns with {} unique values\n'.format(value))

    # Numerical tansformations
    def lognumeric(self):
        self.target.loc[:, 'SalePriceLog'] = np.log1p(self.target['SalePrice'])
        self.df_total[self.numeric_cols] = np.log1p(self.df_total[self.numeric_cols])
        print('5. Applied logaritmic + 1 transformation to numeric values\n')

    # Train and test scaling
    def scale(self):
        # We can reshape total data into train and test
        df_train_prep = self.df_total[:self.df_train.shape[0]]
        df_test_prep = self.df_total[self.df_train.shape[0]:]

        # Updating index
        df_train_prep.reset_index(drop=True, inplace=True)
        df_test_prep.reset_index(drop=True, inplace=True)

        # Scaler
        scaler = StandardScaler()
        a = scaler.fit_transform(df_train_prep)
        b = scaler.transform(df_test_prep)

        # After scaling we need to transform it again to dataframe
        self.df_train_scal = pd.DataFrame(a, columns=df_train_prep.columns).drop(['Id'], axis=1)
        self.df_test_scal = pd.DataFrame(b, columns=df_test_prep.columns).drop(['Id'], axis=1)
        print('6. Applied Standart Scaler\n')

    # Defining our Gradient Boosting model with grid search
    def defmodel(self):
        # We define X as train data and Y as target
        X = self.df_train_scal
        Y = self.target['SalePriceLog']
        model = GradientBoostingRegressor()
        params = {'learning_rate': (0.01, 0.05, 0.1, 0.15),
                    'n_estimators': (100, 200, 300),
                    'max_depth': (3, 5, 10),
                    'min_impurity_decrease': (0, 0.01)
                    }
        init = time.time()
        self.mod = GridSearchCV(estimator=model, n_jobs=16, param_grid=params, cv=5, verbose=1)
        # Fitting our created model with train data
        self.mod.fit(X, Y)
        print('\nGenerated {}\n'.format(self.mod.best_estimator_))
        fin = time.time()
        # Show Grid Search time
        print('Total time: {}\n'.format(fin-init))
        # Saving our best model to an object
        joblib.dump(self.mod.best_estimator_, './Model/house_price_model.pkl')
        print('7. Saved house_price_model.pkl file in Model directory\n')

    # Predicting SalePrice from test
    def predresults(self):
        # Predicting results from test data and transform then again to natural numbers
        final_pred_log = self.mod.predict(self.df_test_scal)
        self.final_pred = np.expm1(final_pred_log)

    # Generate csv submissions file with Id and SalePrice
    def makecsv(self, filename):
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
        # Generamos el archivo que deberá ser subido a la competición
        submission = pd.DataFrame({'Id': self.df_test['Id'], 'SalePrice': self.final_pred})
        print('Showing first 5 predicted values\n')
        print(submission.head())
        submission.to_csv(r'./Submissions/' + filename, index=False)
        print('\n8. File {} generated at Submissions directory\n'.format(filename))

