# HAR-Comparison-Privacy

This Repository contains the code for experimenting with 3 different HAR datasets and checking if they can be compared with each other. Also checking if the data can be used for privacy preserving machine learning(federated learning).

## Datasets

The datasets used are:
- WISDM dataset, more information can be found in data/activity_recognition_wisdm/README.md
- mHealth dataset, more information can be found in data/activity_recognition_mHealth/README.md
- Senior Citizens dataset, more information can be found in data/activity_recognition_senior_citizens/README.md

## Preprocessing
Each dataset is preprocessed in the same way. The preprocessing steps are:
- Dropping not needed columns
- Renaming columns to be more readable and uniform
- Removing rows with missing values, NaN values
- Lowpass filtering the data
- Extracting windows and extracting statistical features from the windows
    - mean
    - std
    - min
    - max
    - range
    - iqr
    - kurtosis
    - skewness
    - rms

## Experiments
On each dataset is performed simple ML pipeline with 3 different classifiers:
- Random Forest
- XGBoost
- LightGBM

The pipeline is:
- Splitting the data into train and test set(80/20)
- Training the classifier on the train set
- Evaluating the classifier on the test set
- Getting accuracy, f1-score macro, confusion matrix, y_true, y_pred, train subjects, test subjects
- Results are saved on local mlflow server

## Results
To see the results navigate to the src/pipelines folder and run the following command:
```
mlflow server
```
Then navigate to http://127.0.0.1:5000/ and you will see the results for each experiment.

## Requirements
To use the code you need:
- Python >= 3.9

To install the requirements run the following command:
```
pip install -r requirements.txt
```

## Structure
The structure of the repository is:
```
.
├── data                                            # Contains the raw data(datasets)
│   ├── activity_recognition_mHealth                # Contains the mHealth raw data + explanation
│   ├── activity_recognition_senior_citizens        # Contains the Senior Citizens raw data + explanation
│   └── activity_recognition_wisdm                  # Contains the WISDM raw data + explanation
├── notebooks                                       # Contains the notebooks(mostly for preprocessing)
│   ├── mHealth_preprocessing.ipynb                 # Contains the mHealth preprocessing
│   ├── Senior_Citizens_preprocessing.ipynb         # Contains the Senior Citizens preprocessing
│   └── WISDM_preprocessing.ipynb                   # Contains the WISDM preprocessing
├── processed_data                                  # Contains the preprocessed data (ignored by git, only locally)
│   ├── activity_recognition_mHealth                # Contains the preprocessed mHealth data
│   ├── activity_recognition_senior_citizens        # Contains the preprocessed Senior Citizens data
│   └── activity_recognition_wisdm                  # Contains the preprocessed WISDM data
├── src                                             # Contains the source code                               
│   ├── figures                                     # Contains the figures for each experiment
│   │   ├── mHealth_80_20                           # Contains the figures for mHealth 80/20 experiment
│   │   ├── senior_citizens_80_20                   # Contains the figures for Senior Citizens 80/20 experiment
│   │   └── wisdm_80_20                             # Contains the figures for WISDM 80/20 experiment
│   ├── pipelines                                   # Contains the code for the experiments(pipelines)
│   │   ├── mlruns                                  # Contains the results for each experiment
│   │   ├── mHealth_train_test_val.py               # Contains the mHealth train/test validation pipeline
│   │   ├── Senior_Citizens_train_test_val.py       # Contains the Senior Citizens train/test validation pipeline
│   │   └── WISDM_train_test_val.py                 # Contains the WISDM train/test validation pipeline
│   └── utils                                       # Contains the utility scripts
│       ├── feature_extraction.py                   # Contains the feature extraction functions
│       ├── mlflow_tracking_experiment.py           # Contains the mlflow tracking functions
│       ├── preprocessing.py                        # Contains the preprocessing functions
│       ├── validation.py                           # Contains the validation functions
│       └── visualization.py                        # Contains the visualization functions
├── .gitignore                                      # Contains the files to be ignored by git
├── README.md                                       # Contains the README file
└── requirements.txt                                # Contains the requirements for the project
```