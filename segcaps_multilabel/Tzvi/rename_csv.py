import glob
from os import path
from shutil import copy

# rename(r'file path\OLD file name.file type',r'file path\NEW file name.file type')

csv_path = r'/home/tzvi/Documents/master/medical_deep_learning/segcaps_results/segcapsr3_heart/split_0'
files = glob.glob(path.join(csv_path, '*.csv'))

for file_i in files:
    file_name = path.split(file_i)[-1]
    file_index = file_name.split('_')[-1].split('s')[0]
    new_name = file_name.replace(file_index, str(int(file_index)+3))
    copy(file_i, new_name)
