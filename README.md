# localstreet

This is a template library for getting the Kaggle Jane Street competition code running locally. The standing assumption is that you are *already* running inside the Kaggle python docker image locally (some notes on how to that are [here](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199214#1101078)).


# Makefile


- `make data` - sets up paths and pulls kaggle data; you need to have a valid `~./kaggle/kaggle.json` or environment variables set as per [Kaggle CLI docs](https://github.com/Kaggle/kaggle-api#api-credentials). This `make` command also executes `export PYTHONPATH=$PYTHONPATH:$PWD/input` so that Python can import the `janestreet` module.



