from utils import load_transformed_tables_from_csv
from features import create_features
import pickle
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from etl import get_engine, get_features, load_config
from os import path
import os

f1_scorer = make_scorer(f1_score, pos_label=1)

def train_xgb_model(train_set, train_labels, n_iter=80, n_splits=3, scoring=f1_scorer):
    """
    Train XGB model with hyperparameter tuning using Randomized Grid search with pre-defined parameter grid.
    Stratified Kfold cross validation is used to select the best model. Metric is F1 score by default.

    :param train_set: pandas.DataFrame object, train set containing features
    :param train_labels: pandas.Series object, train labels
    :param n_iter: int, Number of samples drawn from parameter grid
    :param n_splits: int, Number of splits used in Stratified KFold cross validation
    :param scoring: callable or string, scoring metric used in cross validation
    """
    param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [70, 100, 150],
        'scale_pos_weight': [1, 2, 4, (train_labels==-1).sum()/(train_labels==1).sum()]
    }

    clf = XGBClassifier()

    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=n_iter,
                                    n_jobs=10, verbose=2, cv=StratifiedKFold(n_splits=n_splits),
                                    scoring=f1_scorer, refit=True, random_state=42)

    rs_clf.fit(train_set, train_labels)

    return rs_clf

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model.best_estimator_, f)

if __name__ == '__main__':
    database_config = load_config('../misc/database_config.yaml')[0]

    engine = get_engine(database_config)

    processed_data, processed_labels = get_features(engine)
    samples_with_labels = ~processed_labels.isna()
    processed_data = processed_data[samples_with_labels]
    processed_labels = processed_labels[samples_with_labels]

    model = train_xgb_model(processed_data, processed_labels)

    model_path = '../artifacts'

    if not path.exists(model_path):
        os.makedirs(model_path)


    save_model(model, path.join(model_path, 'model.pkl'))