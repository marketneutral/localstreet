data:
	mkdir -p input
	kaggle competitions download -c jane-street-market-prediction -p input
	unzip -o input/*.zip -d input
	export PYTHONPATH=$PYTHONPATH:$(pwd)/input	
