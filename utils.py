import uuid
import data
import numpy as np
import keras_tuner
import pandas as pd
import tensorflow as tf
from learning import HyperModel
from scipy.stats import linregress
from sklearn.model_selection import KFold
from keras_tuner import BayesianOptimization
from keras_tuner_cv.inner_cv import inner_cv


def get_data(dataset_opt, fix_uptakes, omics, use_deviations, use_flux_mask=False):
    """
    Function used to load the data, according to provided specifications

    Args:
        dataset_opt (str): The name of the dataset to use. Refers to info in data.py (e.g. 'ishii', 'holm')
        fix_uptakes (bool): A flag to signal the uptake fixing (True-yes, False-No)
        omics (list): A list of strings with the omics to be used (e.g. ['transcriptomics','proteomics'])
        use_deviations (bool): A flag to signal the use of standard deviations available from experimental measurements
        use_flux_mask (bool): A flag to signal the use of a flux signal mask (only relevant for deep learning models)

    Returns:
        X (DataFrame): The features (proteomics and/or transcriptomics derived)
        y (DataFrame): The target flux value vectors
        flux_mask (list): A list of 1's and -1's reflecting flux direction
    """
    inputs_dict = data.configs

    # Settings and Data
    prior_fluxes = inputs_dict[dataset_opt]["prior_fluxes"]
    if use_flux_mask:
        flux_mask = inputs_dict[dataset_opt]["flux_mask"][fix_uptakes]
    else:
        flux_mask = None

    # Check the use of standard deviations data
    if use_deviations:
        omics = omics + [omic+"_std" for omic in omics]
    
    # Build X
    dataframes = []
    for omic in omics:
        df = pd.read_csv(inputs_dict[dataset_opt][omic], index_col=0).T
        df.columns = [omic+'_'+c for c in df.columns]
        dataframes.append(df)        
    X = pd.concat(dataframes, axis=1)

    # Build y
    y = pd.read_csv(inputs_dict[dataset_opt]["fluxomics"], index_col=0).T

    # Include Prior Knowledge
    if fix_uptakes:
        X = pd.concat([X,y[prior_fluxes]], axis=1)
        y.drop(y[prior_fluxes], axis=1, inplace=True)

    return X, y, flux_mask


def nn_hyperopt(X_train, y_train, name, batch_size, max_trials, search_epochs):
    """
    Prepare a suitable Neural Network model for the given training data (includes CV style hyperparameterization)

    Args:
        X_train (np.array): The training deatures to be used for hyperparameterization
        y_train (np.array): The training targets to be used for hyperparameterization
        name (str): A name that can be used for serialization purposes
        batch_size (int): Batch size for network training
        max_trials (int): Maximum number for Bayesian Optimization trials
        search_epochs (int): How many epochs to train a model with a given set of hyperparameters

    Returns:
        A model instance built with the best set of hyperparameters found
    """
    tuner = inner_cv(BayesianOptimization)(
        HyperModel(y_train.shape[1], False),
        KFold(n_splits=5, random_state=12345, shuffle=True),
        objective=keras_tuner.Objective("val_loss", direction="min"),
        project_name="Model Optimization",
        directory='tuners/tuner_'+name+'_'+str(uuid.uuid1()),
        seed=12345,
        overwrite=True,
        max_trials=max_trials,
        )

    tuner.search(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        validation_batch_size=batch_size,
        epochs=search_epochs,
        verbose=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]   
    
    return tuner.hypermodel.build(best_hps)
    

def adapt_val_to_train(A, B):
    """
    Adapt dataset B to the same shape and order of dataset A (useful for external validation)
    
    Args:
        A (DataFrame): A DataFrame used for training whose shape we want to reflect on B
        B (DataFrame): A DataFrame used for validation whose shape we want to change to reflect that of A

    Returns:
        merged (DataFrame): A modified version of B
    """
    common_columns = list(A.columns.intersection(B.columns))

    # Create a new DataFrame with common columns from B
    merged = B[common_columns]

    # Append columns from A that are not in B, temporarily filling missing values with zeros
    for col in A.columns:
        if col not in merged.columns:
            merged[col] = 0

    # Reorder columns to match the order in A
    merged = merged[A.columns]

    # Filter out columns that are all zeros
    merged = merged.loc[:, (merged != 0).any(axis=0)]

    return merged


def replace_strings_with_dictionary_values(lst, dictionary):
    """
    Auxiliary function to clean up the names of fluxes

    Args:
        lst (list): A list of strings
        dictionary (dict): A dictionary with the pairs (k,v) of values we want to replace (k) and the value we assign as substitute (v)

    Returns:
        replaced_list (list): A modified version of lst with the specified values replaced.
    """
    replaced_list = []
    for item in lst:
        if item in dictionary:
            replaced_list.append(dictionary[item])
        else:
            replaced_list.append(item)
    return replaced_list


def mae_metric(true,pred,axis=1):
    """
    A function to calculate the MAE between two matrixes

    Args:
        matrix1 (np.array): Typically, the matrix with the actual measured values (true values)
        matrix2 (np.array): Typically, the matrix with the predicted values (output from model)

    Returns:
        A vector with pairwise MAE scores between matrix1 and matrix2 (can be averaged to obtain single value)
    """
    return np.mean(np.abs(true-pred), axis=axis)


def rmse_metric(true,pred, axis=1):
    """
    A function to calculate the RMSE between two matrixes

    Args:
        matrix1 (np.array): Typically, the matrix with the actual measured values (true values)
        matrix2 (np.array): Typically, the matrix with the predicted values (output from model)

    Returns:
        A vector with pairwise RMSE scores between matrix1 and matrix2 (can be averaged to obtain single value)
    """
    return np.sqrt(np.mean(np.square(true-pred), axis=axis))


def ne_metric(true,pred, axis=1):
    """
    A function to calculate the NE between two matrixes

    Args:
        matrix1 (np.array): Typically, the matrix with the actual measured values (true values)
        matrix2 (np.array): Typically, the matrix with the predicted values (output from model)

    Returns:
        A vector with pairwise NE scores between matrix1 and matrix2 (can be averaged to obtain single value)
    """
    return np.nan_to_num(np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis), posinf=0, neginf=0, nan=0)


def r2_metric(true,pred):
    """
    A function to calculate the R² between two matrixes (using scipy's linregress)

    Args:
        matrix1 (np.array): Typically, the matrix with the actual measured values (true values)
        matrix2 (np.array): Typically, the matrix with the predicted values (output from model)

    Returns:
        A vector with pairwise R² scores between matrix1 and matrix2 (can be averaged to obtain single value)
    """
    m, n = true.shape
    r_squared_scores = np.zeros((m,))

    for i in range(m):
        y_true = true[i, :]
        y_pred = pred[i, :]
        try:
            slope, intercept, r, p, se = linregress(y_true, y_pred)
        except:
            r_squared_scores[i] = 0
            continue
        r_squared_scores[i] = r**2

    return r_squared_scores


def custom_sort(element):
    """
    To print the outputs and make sure Holm's results come at the end
    """
    if 'h' in element.split('_')[1][0]:
        return (1, element)
    else:
        return (0, element)