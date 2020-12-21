data:
	mkdir -p input
	kaggle competitions download -c jane-street-market-prediction -p input
	unzip -o input/*.zip -d input
	mkdir -p input/jane-street-market-prediction
	mv input/train.csv input/jane-street-market-prediction
	mv input/features.csv input/jane-street-market-prediction
	export PYTHONPATH=$PYTHONPATH:$PWD/input	
