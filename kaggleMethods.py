import time
from tabulate import tabulate
import zipfile
import os

# Setting enviroment variables manually
__autUSer = {"username": '' , "key": ''}
os.environ['KAGGLE_USERNAME'] = __autUSer.get('username')
os.environ['KAGGLE_KEY'] = __autUSer.get('key')

# API Kaggle use to access functions, to install -> !pip install kaggle
# Installation info: https://www.kaggle.com/docs/api#getting-started-installation-&-authentication
from kaggle.api.kaggle_api_extended import KaggleApi


# Class that allows us to create instances without using json file
class KaggleMethods(object):

    # Constructor to initialize api and submissions
    def __init__(self, comp):
        self.__api = KaggleApi()
        self.__api.authenticate()
        self.__listSubmissions = []
        self.__sizeList = len(self.__listSubmissions)
        self.__compName: str = comp

    # Downloads competition files and unzip them at dwlPath
    def getFiles(self):
        dwlpath = './Kaggle_data/'
        self.__api.competition_download_files(competition=self.__compName, path=dwlpath)
        with zipfile.ZipFile(dwlpath + self.__compName + '.zip', 'r') as zipref:
            zipref.extractall(dwlpath)

    # Updates competition submissions
    def __setList(self):
        submissions = self.__api.competitions_submissions_list(self.__compName)
        for i in submissions:
            dictres = i
            dictres['date'] = dictres.get('date').replace('T', ' ')[:16]
            values = [dictres[k] for k in ['ref', 'fileName', 'date', 'description', 'publicScore', 'teamName']]
            if values not in self.__listSubmissions:
                self.__listSubmissions.append(values)
                self.__sizeList+1

    # Shows competition submissions
    def getList(self):
        if len(self.__listSubmissions) == 0:
            self.__setList()
        table = [['Reference', 'File Name', 'Date', 'Description', 'Public Score', 'Team Name']]
        for i in self.__listSubmissions:
            table.append(i)
        return print(tabulate(table, tablefmt='fancy_grid'))

    # Uploads a new prediction file
    def submitPred(self, file: str, description: str):
        self.__api.competition_submit(file, description, self.__compName)
        time.sleep(5)
        self.__setList(self.__compName)
