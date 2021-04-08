import os

path = '/Users/tanyangwenjian/Downloads/105_classes_pins_dataset/'
for root in os.listdir(path):
    tmp = root.split(sep='_')
    os.rename(path+root, path+tmp[1])