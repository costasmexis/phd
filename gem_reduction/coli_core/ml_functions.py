import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC


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

def train_svm(X, y):

    parameters = {
        'C' : [.01, .05, .1, .5, 1, 10, 100],
        'gamma' : [.01, .1, .5, 1, 2, 5, 10],
        'kernel' : ['rbf', 'linear']
    }    

    estimator = SVC()

    # --------------- RandomizedSearchCV ------------------- #
    tune_model = RandomizedSearchCV(estimator, parameters, n_iter=20, random_state=2, 
                    verbose=10, cv=kf, scoring='roc_auc', n_jobs=-1)

    tune_model.fit(X, y)

    score = cross_val_score(tune_model, X, y, cv=kf)

    return tune_model.best_estimator_, score


def validation_classification(model, X_test, y_test):
    pred = model.predict(X_test)
    print('ROC-AUC =', roc_auc_score(y_test, pred))
    print('ACCURACY =', accuracy_score(y_test, pred))
    print('F1 =', f1_score(y_test, pred))
    return