import yaml
import pickle
from os import path
from etl import load_config, get_engine, get_features
import sqlalchemy


def patrol(user_id, database_config='../misc/database_config.yaml',
                patrol_config='../misc/patrol_config.yaml', model_file='../artifacts/model.pkl'):
    """
    Takes user_id, suggests an action.

    It may be wise to lock user when we are highly certain that the user is a fraudster. To do this, we may aim to
    maximize our utility. I defined a cost that denotes the loss of company when we didn't lock a fraudster but only
    alert the agent. and another cost that denotes the loss of company when we locked a normal user despite that
    they are innocent. The algorithm determines the action based on the losses given by the company.

    :param user_id:
    :param database_config: str or sqlalchemy.engine.base.Engine: If str, a connection will be initialized. If it is
     a sqlalchemy Engine object, it will be directly used for retrieving data from database.
    :param patrol_config: str, the path where the cost values are stored
    :param model_file: str, the path where the model is stored
    :return: action: str, {LOCK_USER, ALERT_AGENT, NONE}
    """
    if not path.exists(model_file):
        raise Exception("Model file could not be found.")

    if type(database_config) == sqlalchemy.engine.base.Engine:
        engine = database_config
    else:
        database_config = load_config(database_config)[0]
        engine = get_engine(database_config)

    user_features, _ = get_features(engine, user_id)

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    prob_being_fraudster = model.predict_proba(user_features)[0][1]
    prob_being_normal = 1 - prob_being_fraudster

    costs = load_config(patrol_config)

    action_costs = {
        'LOCK_USER': prob_being_normal * costs['cost_of_false_negative'],
        'ALERT_AGENT': prob_being_fraudster * costs['cost_of_false_negative'],
        'NONE': prob_being_fraudster * costs['cost_of_false_positive']
   }

    return max(action_costs, key=action_costs.get)
