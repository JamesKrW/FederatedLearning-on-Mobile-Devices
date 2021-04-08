import os
import pandas as pd

def load_data_csv(path):
    nums = []
    i = 0
    names = []
    for dir in os.listdir(path):
        dirpath = os.path.join(path, dir)
        if os.path.isdir(dirpath) and dir!='.DS_Store':
            names.append({"name":dir, "label":i})
            i += 1
    df = pd.DataFrame(names, columns=['name', 'label'])
    df.to_csv('./namelabel.csv', index=False)

load_data_csv('./train_data_csv')