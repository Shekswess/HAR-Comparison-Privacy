# HAR-Comparison-Privacy

This Repository contains the code for experimenting with 3 different HAR datasets and checking if they can be compared with each other. Also checking if the data can be used for privacy preserving machine learning(federated learning).

## Datasets

The datasets used are:
- WISDM dataset, more information can be found in data/activity_recognition_wisdm/README.md
- mHealth dataset, more information can be found in data/activity_recognition_mHealth/README.md
- Senior Citizens dataset, more information can be found in data/activity_recognition_senior_citizens/README.md

For every dataset there is a README file in the data folder, which contains information about the dataset and how to download it.
For our experiments we only used the Accelerometer data from the datasets.

## Preprocessing
For preprocessing the data we have two different approaches:

1. Preprocessing 1
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

2. Preprocessing 2
Each dataset is preprocessed in the same way. The preprocessing steps are:
- Dropping not needed columns
- Renaming columns to be more readable and uniform
- Concatenating classes to make datasets uniform
- Removing rows with missing values, NaN values
- Lowpass filtering the data
- Downsampling the dataset to 20Hz(if needed)
- Extracting windows and extracting statistical and frequency features from the windows
    Statistical features:
    - mean
    - std
    - min
    - max
    - range
    - iqr
    - kurtosis
    - skewness
    - rms
    Frequency features:
    - mean_freq
    - std_freq
    - max_freq
    - max_freq_mag
    - freq_mean
    - freq_std
    - freq_skew
    - freq_kurtosis
The class labels are uniform for all datasets:
- Still (Not Moving, Standing Still, Sitting, Lying)
- Walking
- Stairs (Walking upstairs, walking downstairs)
- Jogging (Running, Jogging)


## Experiments
We have 3 different experiments:

1. Experiment 1
On each dataset is performed simple ML pipeline with 2 different classifiers and the data is preprocessed with Preprocessing 1. 
The classifiers used are:
- XGBoost
- LightGBM

The pipeline is:
- Splitting the data into train and test set(80/20)
- Training the classifier on the train set
- Evaluating the classifier on the test set
- Getting accuracy, f1-score macro, confusion matrix, y_true, y_pred, train subjects, test subjects
- Results are saved on local mlflow server

The best results are for the WISDM dataset

2. Experiment 2
On each dataset is performed simple ML pipeline with 2 different classifiers and the data is preprocessed with Preprocessing 2.
The classifiers used are:
- XGBoost
- LightGBM

The pipeline is:
- Splitting the data into train and test set(80/20)
- Training the classifier on the train set
- Evaluating the classifier on the test set
- Getting accuracy, f1-score macro, confusion matrix, y_true, y_pred, train subjects, test subjects
- Results are saved on local mlflow server

The best results are for the mHealth dataset

3. Experiment 3
The best and most stable dataset(results)=mHealth from Expreriment 2 is used for this experiment, so the data is preprocessed with Preprocessing 2.
On the dataset is performed federated learning pipeline with XGBoost Boosting Classifier.
The dataset has 10 users and each user is a possible client.

The pipeline is:
- Splitting the data into train and test set(80/20) per user/client where the labels are balanced
- Training the classifier on the train set locally for each user/client
- Evaluating the classifier on the test set locally for each user/client
- Getting accuracy, f1-score macro for each user/client
- Aggregating the results for all users/clients on the server using FedXgbBagging algorithm
- Getting best accuracy, best f1-score macro, average accuracy, average f1-score macro for all users/clients on the server after all rounds are finished/aggregated

The combinations that are tested are:
- number of rounds: 3, 5, 8, 10, 15, 20
- number of clients: 6, 8, 10

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
├── data                                                                                # Contains the raw data(datasets)
│   ├── activity_recognition_mHealth                                                    # Contains the mHealth raw data + explanation
│   ├── activity_recognition_senior_citizens                                            # Contains the Senior Citizens raw data + explanation
│   └── activity_recognition_wisdm                                                      # Contains the WISDM raw data + explanation
├── notebooks                                                                           # Contains the notebooks(mostly for preprocessing)
│   ├── mHealth_preprocessing_less_classes_frequency_features.ipynb                     # Contains the mHealth preprocessing with less classes and frequency features
│   ├── mHealth_preprocessing.ipynb                                                     # Contains the mHealth preprocessing
│   ├── senior_citizens_preprocessing_less_classes_frequency_features.ipynb             # Contains the Senior Citizens preprocessing with less classes and frequency features
│   ├── senior_citizens_preprocessing.ipynb                                             # Contains the Senior Citizens preprocessing
│   ├── WISDM_preprocessing_less_classes_frequency_features.ipynb                       # Contains the WISDM preprocessing with less classes and frequency features
│   └── WISDM_preprocessing.ipynb                                                       # Contains the WISDM preprocessing
├── processed_data                                                                      # Contains the preprocessed data (ignored by git, only locally)
│   ├── activity_recognition_mHealth                                                    # Contains the preprocessed mHealth data
│   ├── activity_recognition_mHealth_less_classes_frequency_features                    # Contains the preprocessed mHealth data with less classes and frequency features
│   ├── activity_recognition_senior_citizens                                            # Contains the preprocessed Senior Citizens data
│   ├── activity_recognition_senior_citizens_less_classes_frequency_features            # Contains the preprocessed Senior Citizens data with less classes and frequency features
│   ├── activity_recognition_wisdm_less_classes_frequency_features                      # Contains the preprocessed WISDM data with less classes and frequency features
│   └── activity_recognition_wisdm                                                      # Contains the preprocessed WISDM data
├── src                                                                                 # Contains the source code
│   ├── federated_xgboost                                                               # Contains the federated xgboost server and client code
│   │   ├── client.py                                                                   # Contains the federated xgboost client code
│   │   └── server.py                                                                   # Contains the federated xgboost server code                           
│   ├── figures                                                                         # Contains the figures for each experiment
│   │   ├── federated_xgboost                                                           # Contains the figures for the federated xgboost experiment
│   │   ├── mHealth_80_20                                                               # Contains the figures for mHealth 80/20 experiment
│   │   ├── mHealth_less_classes_frequency_features_80_20                               # Contains the figures for mHealth 80/20 experiment with less classes and frequency features   
│   │   ├── senior_citizens_80_20                                                       # Contains the figures for Senior Citizens 80/20 experiment
│   │   ├── senior_citizens_less_classes_frequency_features_80_20                       # Contains the figures for Senior Citizens 80/20 experiment with less classes and frequency features
│   │   ├── wisdm_80_20                                                                 # Contains the figures for WISDM 80/20 experiment
│   │   └── wisdm_less_classes_frequency_features_80_20                                 # Contains the figures for WISDM 80/20 experiment with less classes and frequency features
│   ├── pipelines                                                                       # Contains the code for the experiments(pipelines)
│   │   ├── mlruns                                                                      # Contains the results for each experiment
│   │   ├── mHealth_federated.py                                                        # Contains the mHealth federated learning pipeline      
│   │   ├── mHealth_less_classes_freq_train_test_val.py                                 # Contains the mHealth train/test validation pipeline with less classes and frequency features
│   │   ├── mHealth_train_test_val.py                                                   # Contains the mHealth train/test validation pipeline
│   │   ├── senior_citizens_less_classes_freq_train_test_val.py                         # Contains the Senior Citizens train/test validation pipeline with less classes and frequency features
│   │   ├── senior_citizens_train_test_val.py                                           # Contains the Senior Citizens train/test validation pipeline
│   │   ├── WISDM_less_classes_freq_train_test_val.py                                   # Contains the WISDM train/test validation pipeline with less classes and frequency features
│   │   └── WISDM_train_test_val.py                                                     # Contains the WISDM train/test validation pipeline
│   └── utils                                                                           # Contains the utility scripts
│       ├── feature_extraction.py                                                       # Contains the feature extraction functions
│       ├── mlflow_tracking_experiment.py                                               # Contains the mlflow tracking functions
│       ├── preprocessing.py                                                            # Contains the preprocessing functions
│       ├── validation.py                                                               # Contains the validation functions
│       └── visualization.py                                                            # Contains the visualization functions
├── .gitignore                                                                          # Contains the files to be ignored by git
├── README.md                                                                           # Contains the README file
└── requirements.txt                                                                    # Contains the requirements for the project
```