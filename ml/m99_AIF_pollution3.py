# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from keras import regularizers


# Load the data
path='c:/study/_data/AIFac_pollution/'
save_path= 'c:/study/_save/AIFac_pollution/'


train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
