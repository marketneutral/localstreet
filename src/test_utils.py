import janestreet
from utils import read_train_data

def test_pred_loop():
    env = janestreet.make_env() # initialize the environment
    iter_test = env.iter_test() # an iterator which loops over the test set

    for i, (test_df, sample_prediction_df) in enumerate(iter_test):
        sample_prediction_df.action = 0 #make your 0/1 prediction here
        env.predict(sample_prediction_df)
        if i % 100 == 0:
            print(f'Test set record {i}: {test_df}')
            
        if i == 300:
            break

    return True

def test_read_data():
    df = read_train_data()
    return True
