# 1. 데이터
path = 'd:/study_data/_data/gas/'  
save_path = 'd:/study_data/_save/gas/'


train_csv = pd.read_csv(path + 'train_data.csv', 
                        index_col=0) 

test_csv = pd.read_csv(path + 'test_data.csv',
                       index_col=0) 





#submission.csv 만들기
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(y_submit)
# print(y_submit.shape)
y_submit = np.argmax(y_submit, axis=1)
print(y_submit.shape)
y_submit += 3
submission['quality'] = y_submit
# print(submission)

path_save = 'd:/study_data/_save/gas/' 
submission.to_csv(path_save + 'gas_01.csv')