from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# PCA를 사용하여 0.95, 0.99, 0.999, 1.0 이상의 variance ratio를 가지는 feature 개수를 구함
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d95 = np.argmax(cumsum >= 0.95) + 1
d99 = np.argmax(cumsum >= 0.99) + 1
d999 = np.argmax(cumsum >= 0.999) + 1
d100 = np.argmax(cumsum == 1.0) + 1

# 각각의 feature 개수에 대해 RandomForestClassifier 모델의 정확도 계산
for d in [d95, d99, d999, d100]:
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    
    # Define the hyperparameters to search
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    # Create the RandomSearchCV object
    random_search = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                                       n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    
    # Fit the random search model
    random_search.fit(x_train, y_train)
    
    # Print the best parameters and accuracy score
    print(f"PCA {d} - Best Parameters: {random_search.best_params_}")
    print(f"PCA {d} - Best Accuracy: {random_search.best_score_:.4f}")
