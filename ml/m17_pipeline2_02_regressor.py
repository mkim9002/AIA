from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
import pandas as pd

path_d = './_data/ddarung/'
path_k = './_data/kaggle_bike/'

ddarung = pd.read_csv(path_d + 'train.csv', index_col=0).dropna()
kaggle = pd.read_csv(path_k + 'train.csv', index_col=0).dropna()

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_california_housing, load_diabetes, ddarung, kaggle]
scaler_list = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler()]
classifier_list = [SVC(), RandomForestClassifier(), DecisionTreeClassifier()]
regressor_list = [RandomForestRegressor(), DecisionTreeRegressor()]

for i in range(len(data_list)):
    if i<4:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_acc = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in classifier_list:
                model = make_pipeline(j, k)
                model.fit(x_train, y_train)
                if max_acc < accuracy_score(y_test, model.predict(x_test)):
                    max_acc = accuracy_score(y_test, model.predict(x_test))
                    max_model = type(k).__name__
                    max_scaler = type(j).__name__
                print('\n', data_list[i].__name__, type(k).__name__, type(j).__name__, '\nresult : ', model.score(x_test, y_test), '\nacc : ', accuracy_score(y_test, model.predict(x_test)))
        print('\n', data_list[i].__name__, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_acc : ', max_acc)
    elif 4<=i<6:
        x, y = data_list[i](return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in regressor_list:
                model = make_pipeline(j, k)
                model.fit(x_train, y_train)
                if max_r2 < r2_score(y_test, model.predict(x_test)):
                    max_r2 = r2_score(y_test, model.predict(x_test))
                    max_model = type(k).__name__
                    max_scaler = type(j).__name__
                print('\n', data_list[i].__name__, type(k).__name__, type(j).__name__,'\nresult : ', model.score(x_test, y_test), '\nr2 : ', r2_score(y_test, model.predict(x_test)))
        print('\n', data_list[i].__name__, '\nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)
    elif i==6:
        x = data_list[i].drop(['count'], axis=1)
        y = data_list[i]['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in regressor_list:
                model = make_pipeline(j, k)
                model.fit(x_train, y_train)
                if max_r2 < r2_score(y_test, model.predict(x_test)):
                    max_r2 = r2_score(y_test, model.predict(x_test))
                    max_model = type(k).__name__
                    max_scaler = type(j).__name__
                print('\n', 'ddarung', type(k).__name__, type(j).__name__, '\nresult : ', model.score(x_test, y_test), '\nr2 : ', r2_score(y_test, model.predict(x_test)))
        print('\n ddarung \nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)
    elif i==7:
        x = data_list[i].drop(['casual', 'registered', 'count'], axis=1)
        y = data_list[i]['count']
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)
        max_r2 = 0
        max_model = ''
        max_scaler = ''
        for j in scaler_list:
            for k in regressor_list:
                model = make_pipeline(j, k)
                model.fit(x_train, y_train)
                if max_r2 < r2_score(y_test, model.predict(x_test)):
                    max_r2 = r2_score(y_test, model.predict(x_test))
                    max_model = type(k).__name__
                    max_scaler = type(j).__name__
                print('\n', 'kaggle', type(k).__name__, type(j).__name__, '\nresult : ', model.score(x_test, y_test), '\nr2 : ', r2_score(y_test, model.predict(x_test)))
        print('\n kaggle \nmax_model : ', max_model, '\nmax_scaler : ', max_scaler, '\nmax_r2 : ', max_r2)
        
        
        
#          load_iris SVC MinMaxScaler 
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris RandomForestClassifier MinMaxScaler 
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris DecisionTreeClassifier MinMaxScaler
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris SVC StandardScaler
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris RandomForestClassifier StandardScaler 
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris DecisionTreeClassifier StandardScaler
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris SVC MaxAbsScaler
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris RandomForestClassifier MaxAbsScaler 
# result :  0.9333333333333333
# acc :  0.9333333333333333

#  load_iris DecisionTreeClassifier MaxAbsScaler
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris SVC RobustScaler
# result :  0.9111111111111111
# acc :  0.9111111111111111

#  load_iris RandomForestClassifier RobustScaler 
# result :  0.9555555555555556
# acc :  0.9555555555555556

#  load_iris DecisionTreeClassifier RobustScaler
# result :  0.9333333333333333
# acc :  0.9333333333333333

#  load_iris
# max_model :  SVC
# max_scaler :  MinMaxScaler
# max_acc :  0.9555555555555556

#  load_breast_cancer SVC MinMaxScaler
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer RandomForestClassifier MinMaxScaler 
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer DecisionTreeClassifier MinMaxScaler
# result :  0.9532163742690059
# acc :  0.9532163742690059

#  load_breast_cancer SVC StandardScaler
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer RandomForestClassifier StandardScaler 
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer DecisionTreeClassifier StandardScaler
# result :  0.935672514619883
# acc :  0.935672514619883

#  load_breast_cancer SVC MaxAbsScaler
# result :  0.9883040935672515
# acc :  0.9883040935672515

#  load_breast_cancer RandomForestClassifier MaxAbsScaler 
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer DecisionTreeClassifier MaxAbsScaler
# result :  0.9415204678362573
# acc :  0.9415204678362573

#  load_breast_cancer SVC RobustScaler
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer RandomForestClassifier RobustScaler 
# result :  0.9824561403508771
# acc :  0.9824561403508771

#  load_breast_cancer DecisionTreeClassifier RobustScaler
# result :  0.935672514619883
# acc :  0.935672514619883

#  load_breast_cancer
# max_model :  SVC
# max_scaler :  MaxAbsScaler
# max_acc :  0.9883040935672515

#  load_digits SVC MinMaxScaler 
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_digits RandomForestClassifier MinMaxScaler 
# result :  0.9629629629629629
# acc :  0.9629629629629629

#  load_digits DecisionTreeClassifier MinMaxScaler
# result :  0.8611111111111112
# acc :  0.8611111111111112

#  load_digits SVC StandardScaler 
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_digits RandomForestClassifier StandardScaler 
# result :  0.9666666666666667
# acc :  0.9666666666666667

#  load_digits DecisionTreeClassifier StandardScaler
# result :  0.8555555555555555 
# acc :  0.8555555555555555

#  load_digits SVC MaxAbsScaler 
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_digits RandomForestClassifier MaxAbsScaler 
# result :  0.9666666666666667
# acc :  0.9666666666666667

#  load_digits DecisionTreeClassifier MaxAbsScaler
# result :  0.8462962962962963 
# acc :  0.8462962962962963

#  load_digits SVC RobustScaler 
# result :  0.9592592592592593
# acc :  0.9592592592592593

#  load_digits RandomForestClassifier RobustScaler 
# result :  0.9722222222222222
# acc :  0.9722222222222222

#  load_digits DecisionTreeClassifier RobustScaler
# result :  0.8481481481481481 
# acc :  0.8481481481481481

#  load_digits
# max_model :  SVC
# max_scaler :  MinMaxScaler
# max_acc :  0.9814814814814815

#  load_wine SVC MinMaxScaler
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_wine RandomForestClassifier MinMaxScaler 
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_wine DecisionTreeClassifier MinMaxScaler
# result :  0.9629629629629629
# acc :  0.9629629629629629

#  load_wine SVC StandardScaler
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_wine RandomForestClassifier StandardScaler 
# result :  0.9629629629629629
# acc :  0.9629629629629629

#  load_wine DecisionTreeClassifier StandardScaler
# result :  0.9444444444444444
# acc :  0.9444444444444444

#  load_wine SVC MaxAbsScaler
# result :  1.0
# acc :  1.0

#  load_wine RandomForestClassifier MaxAbsScaler 
# result :  1.0
# acc :  1.0

#  load_wine DecisionTreeClassifier MaxAbsScaler
# result :  0.9629629629629629
# acc :  0.9629629629629629

#  load_wine SVC RobustScaler
# result :  0.9814814814814815
# acc :  0.9814814814814815

#  load_wine RandomForestClassifier RobustScaler 
# result :  0.9629629629629629
# acc :  0.9629629629629629

#  load_wine DecisionTreeClassifier RobustScaler
# result :  0.9444444444444444
# acc :  0.9444444444444444

#  load_wine
# max_model :  SVC
# max_scaler :  MaxAbsScaler
# max_acc :  1.0

#  fetch_california_housing RandomForestRegressor MinMaxScaler 
# result :  0.8124554025617641
# r2 :  0.8124554025617641

#  fetch_california_housing DecisionTreeRegressor MinMaxScaler 
# result :  0.6340096903280028
# r2 :  0.6340096903280028

#  fetch_california_housing RandomForestRegressor StandardScaler 
# result :  0.8108246083176074
# r2 :  0.8108246083176074

#  fetch_california_housing DecisionTreeRegressor StandardScaler 
# result :  0.6316215014258812
# r2 :  0.6316215014258812

#  fetch_california_housing RandomForestRegressor MaxAbsScaler 
# result :  0.8112089278536602
# r2 :  0.8112089278536602

#  fetch_california_housing DecisionTreeRegressor MaxAbsScaler 
# result :  0.6376852072270818
# r2 :  0.6376852072270818

#  fetch_california_housing RandomForestRegressor RobustScaler 
# result :  0.8098877159168498
# r2 :  0.8098877159168498

#  fetch_california_housing DecisionTreeRegressor RobustScaler 
# result :  0.6273321386109799
# r2 :  0.6273321386109799

#  fetch_california_housing
# max_model :  RandomForestRegressor
# max_scaler :  MinMaxScaler
# max_r2 :  0.8124554025617641

#  load_diabetes RandomForestRegressor MinMaxScaler 
# result :  0.42464546276545334
# r2 :  0.42464546276545334

#  load_diabetes DecisionTreeRegressor MinMaxScaler
# result :  0.00183631078998292
# r2 :  0.00183631078998292

#  load_diabetes RandomForestRegressor StandardScaler 
# result :  0.4300375187135169
# r2 :  0.4300375187135169

#  load_diabetes DecisionTreeRegressor StandardScaler
# result :  -0.013472591432998504
# r2 :  -0.013472591432998504

#  load_diabetes RandomForestRegressor MaxAbsScaler 
# result :  0.4373055344686916
# r2 :  0.4373055344686916

#  load_diabetes DecisionTreeRegressor MaxAbsScaler
# result :  -0.04086247819022071
# r2 :  -0.04086247819022071

#  load_diabetes RandomForestRegressor RobustScaler 
# result :  0.4228619631392865
# r2 :  0.4228619631392865

#  load_diabetes DecisionTreeRegressor RobustScaler
# result :  -0.02352300964991949
# r2 :  -0.02352300964991949

#  load_diabetes
# max_model :  RandomForestRegressor
# max_scaler :  MaxAbsScaler
# max_r2 :  0.4373055344686916

#  ddarung RandomForestRegressor MinMaxScaler 
# result :  0.7745282917335958
# r2 :  0.7745282917335958

#  ddarung DecisionTreeRegressor MinMaxScaler
# result :  0.5761204339718271
# r2 :  0.5761204339718271

#  ddarung RandomForestRegressor StandardScaler 
# result :  0.7734678587563497
# r2 :  0.7734678587563497

#  ddarung DecisionTreeRegressor StandardScaler
# result :  0.5465471769916732 
# r2 :  0.5465471769916732

#  ddarung RandomForestRegressor MaxAbsScaler 
# result :  0.7698028507667267
# r2 :  0.7698028507667267

#  ddarung DecisionTreeRegressor MaxAbsScaler
# result :  0.5970676373632565 
# r2 :  0.5970676373632565

#  ddarung RandomForestRegressor RobustScaler 
# result :  0.771971128759946
# r2 :  0.771971128759946

#  ddarung DecisionTreeRegressor RobustScaler
# result :  0.585857097636131 
# r2 :  0.585857097636131

#  ddarung
# max_model :  RandomForestRegressor
# max_scaler :  MinMaxScaler
# max_r2 :  0.7745282917335958

#  kaggle RandomForestRegressor MinMaxScaler 
# result :  0.28433717297721617
# r2 :  0.28433717297721617

#  kaggle DecisionTreeRegressor MinMaxScaler
# result :  -0.11905445079932364
# r2 :  -0.11905445079932364

#  kaggle RandomForestRegressor StandardScaler 
# result :  0.2832157652733419
# r2 :  0.2832157652733419

#  kaggle DecisionTreeRegressor StandardScaler 
# result :  -0.1056163595144044
# r2 :  -0.1056163595144044

#  kaggle RandomForestRegressor MaxAbsScaler 
# result :  0.28611413771275584
# r2 :  0.28611413771275584

#  kaggle DecisionTreeRegressor MaxAbsScaler
# result :  -0.12128634529847981
# r2 :  -0.12128634529847981

#  kaggle RandomForestRegressor RobustScaler 
# result :  0.28971213394511874
# r2 :  0.28971213394511874

#  kaggle DecisionTreeRegressor RobustScaler
# result :  -0.11246429413925063
# r2 :  -0.11246429413925063

#  kaggle
# max_model :  RandomForestRegressor
# max_scaler :  RobustScaler
# max_r2 :  0.28971213394511874