import pickle as pk
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

with open('./db/processed/data_pre_processed.pkl', mode='rb') as f:
    train, y = pk.load(f)

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model_random_forest = RandomForestClassifier(random_state=42)
model_svm = SVC(max_iter=5000, random_state=42)
model_neural_network = MLPClassifier(max_iter=5000, random_state=42)

param_grid_random_forest = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_grid_svc = {
    'C': [0.1, 1, 10], 
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'], 
    'tol': [1e-3, 1e-4]  
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],                          
    'solver': ['adam', 'sgd'],                               
    'alpha': [0.0001, 0.001, 0.01],                        
    'learning_rate_init': [0.001, 0.01],         
}

train_models = {
    'random_forest': [model_random_forest, param_grid_random_forest],
    'svm': [model_svm, param_grid_svc],
    'neural_network': [model_neural_network, param_grid_mlp]
}

results = []

for name, (model, param) in train_models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param, cv=cv_strategy, verbose=2, scoring='f1', n_jobs=-1)
    grid_search.fit(train, y)
    results.append({
        'model': name,
        'score' : grid_search.best_score_
    })

    with open(f'./db/models/{name}_model.pkl', mode='wb') as f:
        pk.dump(grid_search.best_estimator_, f)

results_df = pd.DataFrame(results)
results_df.to_csv('./db/submission/results.csv', sep=',', index=False)