import os
import shutil

def photo_sampling(path):
    for dir in os.listdir(path):
        dirpath=os.path.join(path,dir)
        copydir=os.path.join('./data1/train_data', dir)
        testdir = os.path.join('./data/test_data', dir)
        if os.path.isdir(dirpath):
            count=0
            for img in os.listdir(dirpath):
                if count<=1:
                    if not os.path.exists(copydir):
                        os.makedirs(copydir)
                    count = count + 1
                    oldpath = os.path.join(dirpath, img)
                    newpath = os.path.join(copydir, img)
                    shutil.copy(oldpath, newpath)
                if count>1:
                    continue

photo_sampling('./105_classes_pins_dataset')