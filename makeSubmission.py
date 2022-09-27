import time
from kaggleMethods import KaggleMethods
import sys

__competition = 'house-prices-advanced-regression-techniques'


def __submit(file, obj):
    if isinstance(obj, KaggleMethods):
        while True:
            desc = str(input('\nInsert file description: '))
            print('')
            break
        obj.submitPred(file, desc)
        print('Submitted {} file to {}\n'.format(file, __competition))


def __showresults(obj):
    while True:
        res = str(input('Insert Yes or No to check your competition leaderboard scores: '))
        if res == 'Yes':
            obj.getList()
            break
        elif res == 'No':
            break


def main():
    # Defines file
    if len(sys.argv) == 2:
        arg1 = sys.argv[1]
    else:
        raise Exception('No file detected')
    # Kaggle Api instance
    comp = KaggleMethods(comp=__competition)
    __submit(arg1, comp)
    time.sleep(3)
    # Kaggle submission score
    __showresults(comp)


if __name__ == '__main__':
    main()
