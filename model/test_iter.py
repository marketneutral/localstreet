import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

for i, (test_df, sample_prediction_df) in enumerate(iter_test):
    sample_prediction_df.action = 0 #make your 0/1 prediction here
    env.predict(sample_prediction_df)
    if i % 10000 == 0:
        print(i)
        print(test_df)
