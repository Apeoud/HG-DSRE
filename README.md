# HG-DSRE

This project contains all source code and dataset in ESWC2019 paper "A Hybrid Graph Model for Distant Supervision Relation Extraction"


## dataset 

The dataset was provided by thunlp. 
Please download the dataset from https://drive.google.com/file/d/0B_TJXoToQ4r4QWQtOFRsbEZwTmM/view?usp=sharing.

For more details, please refer to https://github.com/thunlp/PathNRE/tree/master/data.

## source code 
At first, you need run the preprocessing procedure to generate some additional information. By default, we provide entity type and relation path.
```
python src/load.py
```

To train the model, run the following command 
```
python src/train.py 
```

To test the model, run the following command 
```
python src/test.py
```

