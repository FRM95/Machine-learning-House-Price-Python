import time
from kaggleMethods import KaggleMethods
from HousePriceModel import HousePriceModel
import sys
import os


# Generates predictions of gradient boosting model into file
def __generateModel(file):
    # Model instance
    model = HousePriceModel()
    model.checkdata()
    model.defvariables()
    model.imputna()
    model.colstonum()
    model.labencoder()
    model.onehot()
    model.lognumeric()
    model.scale()
    model.defmodel()
    model.predresults()
    model.makecsv(file)

def __downloadData() -> int:
    print('Kaggle_data directory must contain House Price competition files before continue...\n')
    while True:
        print('Do you wish to download House Price Kaggle competition data?\n')
        res = str(input('Insert Yes or No:'))
        if res == 'Yes':
            # Download competition files
            # Kaggle Api instance
            comp = KaggleMethods(comp='house-prices-advanced-regression-techniques')
            print('\nDownloading competition files..\n')
            comp.getFiles()
            time.sleep(2)
            break
            return 0
        elif res == 'No':
            if os.path.exists('./Kaggle_data/test.csv') and os.path.exists('./Kaggle_data/train.csv'):
                break
                return 0
            else:
                print('Error: Files train and test can not be found\n')
        else:
            print('Error: Insert Yes or No:')



def main():
    # CSV prediction file name
    if len(sys.argv) == 2:
        arg1 = sys.argv[1]
    else:
        arg1 = 'submission.csv'
    print('\nWelcome to House Price model generator App\n')
    time.sleep(4)
    # Files download
    __downloadData()
    # Generate Model
    __generateModel(arg1)


if __name__ == '__main__':
    main()
