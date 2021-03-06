import os
import glob
import shutil
import json
import pandas as pd

def continue_train_on(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        usr_in = input('%s already exists. Overwrite? (y/n): ' % path)
        if usr_in.lower() == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            usr_in = input('Continue training this model? (y/n): ')
            if usr_in.lower() == 'y': return True
            else:
                print('Exiting...')
                raise SystemExit

def create_dataset(data_dir, fout='dataset.csv', sort=True):
    cases = [r+'/' for r, d, f in os.walk(data_dir) if not d]
    df = {'Fpath':[], 'MaxIndex':[]}
    for case in cases:
        fpaths = [f for f in glob.glob(case + '/*.DCM', recursive=True)]
        if sort: fpaths.sort(key=lambda fname: fname.split('/')[-1])
        df['Fpath'] += fpaths
        df['MaxIndex'] += [len(fpaths)] * len(fpaths)

    pd.DataFrame(df).to_csv(fout, index=False)

class Params:
    def __init__(self, fparams):
        self.update(fparams)

    def save(self, fparams):
        with open(fparams, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, fparams):
        with open(fparams) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def dict(self):
        return self.__dict__
