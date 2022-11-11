import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV

kf = KFold(n_splits=3, shuffle=True, random_state=4)

def xgb_classifier(X, y):

    parameters = {
        'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
        'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight' : [ 1, 3, 5, 7 ],
        'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
    }    

    estimator = xgb.XGBClassifier()

    # --------------- RandomizedSearchCV ------------------- #
    xgb_clas = RandomizedSearchCV(estimator, parameters, n_iter=20, random_state=2, 
                    verbose=10, cv=kf, scoring='roc_auc', n_jobs=-1)

    xgb_clas.fit(X, y)

    score = cross_val_score(xgb_clas, X, y, cv=kf)

    return xgb_clas.best_estimator_, score
