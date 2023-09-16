# IMPORTS
# ______________________________________________________________________________________________________________________
import sys
 
sys.path.insert(0, 'Prep_Bag_of_Words')
from preprocessing_library import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = 'DSI_output/'
# last year to be used in train set
last_data_year_train = 2014


# SCRIPT
# ______________________________________________________________________________________________________________________
print('READING IN THE DATA')
# read data
healthy = pd.read_csv(data_location + 'intermediate_processed/healthy_bow.csv', index_col=0)
failed = pd.read_csv(data_location + 'intermediate_processed/failed_bow.csv', index_col=0)
# create and store train data used to compute frequencies
train = get_train_corpus(healthy, failed, last_data_year_train)
train.to_csv(data_location + 'intermediate_processed/train_w2v.csv')
