import uuid
import argparse
import learning
import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from keras_tuner_cv.inner_cv import inner_cv
from keras_tuner import BayesianOptimization
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils import get_data, adapt_val_to_train, r2_metric, nn_hyperopt


models_dict = {'dt': DecisionTreeRegressor(),
               'lr': LinearRegression(),
               'rf': RandomForestRegressor(),
               'svm': MultiOutputRegressor(SVR()),
               'xgb': XGBRegressor(),
               'nn': None # assigned upon hyperoptimization
               }


def external_validate(fix_uptakes, use_deviations, name, serialize, batch_size, max_trials, search_epochs):

    X_ishii, y_ishii, _ = get_data('ishii', fix_uptakes, ['transcriptomics'], use_deviations)
    X_holm, y_holm, _ = get_data('holm', fix_uptakes, ['transcriptomics'], use_deviations)

    X_holm = adapt_val_to_train(X_ishii,X_holm)
    y_holm = adapt_val_to_train(y_ishii,y_holm)
    
    X_ishii = X_ishii[X_holm.columns]
    y_ishii = y_ishii[y_holm.columns]

    models_dict['nn'] = nn_hyperopt(X_ishii.values, y_ishii.values, name, batch_size, max_trials, search_epochs)

    for model_name in models_dict.keys():
        model = models_dict[model_name]

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

        # Basic Data Preprocessing
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_ishii)
        y_train = sc_y.fit_transform(y_ishii)
        X_test = sc_X.transform(X_holm)
        y_test = sc_y.transform(y_holm)

        if model_name == 'nn':
            model.fit(X_train, y_train, batch_size=batch_size, epochs=search_epochs,verbose=0)
        else:
            model = model.fit(X_train, y_train)
            
        # Inverse transform targets to extract metrics
        y_pred = model.predict(X_test)
        y_pred = sc_y.inverse_transform(y_pred)
        y_test_aux = sc_y.inverse_transform(y_test)
        
        # Generate metrics
        print('---',model_name+'_'+name,'---')
        print("Average MAE:", round(np.mean(mae(y_test_aux,y_pred).numpy()),3), "±", round(np.std(mae(y_test_aux,y_pred).numpy()),3))
        print("Average RMSE:", round(np.sqrt(np.mean(mse(y_test_aux,y_pred).numpy())),3), "±", round(np.sqrt(np.std(mse(y_test_aux,y_pred).numpy())),3))
        print("Average NE:", round(np.mean(np.nan_to_num(np.linalg.norm(y_test_aux - y_pred, axis=1) / np.linalg.norm(y_test_aux, axis=1), posinf=0, neginf=0, nan=0)),3), "±", round(np.std(np.nan_to_num(np.linalg.norm(y_test_aux - y_pred, axis=1) / np.linalg.norm(y_test_aux, axis=1), posinf=0, neginf=0, nan=0)),3))
        print("Avergae R2:", round(np.mean(r2_metric(y_test_aux, y_pred)), 3), '±', round(np.std(r2_metric(y_test_aux, y_pred)), 3))

        if serialize:
            # Write the predictions to a CSV file
            predictions_matrix = list(zip([0,1,2],y_pred))
            predictions_matrix.sort(key=lambda x: x[0])
            predictions_df = pd.DataFrame([x[1] for x in predictions_matrix], index = X_holm.index).T
            predictions_df.to_csv('results/predictions_'+'holm_'+model_name+'_'+name+'.csv', encoding='utf-8', index=False)


def cross_validate(model_type, dataset, fix_uptakes, omics, use_deviations, name, serialize):
    X, y, _ = get_data(dataset, fix_uptakes, omics, use_deviations)

    model = models_dict[model_type]

    kfold = KFold(n_splits=len(X), shuffle=True)

    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()

    predictions_matrix = []
    rmse_per_fold = []
    mae_per_fold = []
    normalized_error_per_fold = []
    r2_per_fold = []
    fold_no = 1
    inputs, targets = (X.values,y.values)
    for train, test in kfold.split(inputs, targets):

        # Basic Data Preprocessing
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(inputs[train])
        y_train = sc_y.fit_transform(targets[train])
        X_test = sc_X.transform(inputs[test])
        y_test = sc_y.transform(targets[test])

        print('------------------------------------------------------------------------')
        print(f'[{model_type}] Training for fold {fold_no} ...')    
                
        model = model.fit(X_train, y_train)
            
        # Inverse transform targets to extract metrics
        y_pred = model.predict(X_test)
        y_pred = sc_y.inverse_transform(y_pred)
        y_test_aux = sc_y.inverse_transform(y_test)
        predictions_matrix.append((test[0], y_pred[0]))

        # Generate generalization metrics
        print(f'Score for fold {fold_no}: RMSE of {np.sqrt(mse(y_test_aux,y_pred).numpy())}; MAE of {mae(y_test_aux,y_pred).numpy()}; Normalized error of {np.mean(np.nan_to_num(np.linalg.norm(y_test_aux[0] - y_pred[0], axis=0) / np.linalg.norm(y_test_aux[0], axis=0), posinf=0, neginf=0, nan=0))}')
        rmse_per_fold.append(np.sqrt(mse(y_test_aux,y_pred).numpy()))
        mae_per_fold.append(mae(y_test_aux,y_pred).numpy())
        normalized_error_per_fold.append(np.mean(np.nan_to_num(np.linalg.norm(y_test_aux[0] - y_pred[0], axis=0) / np.linalg.norm(y_test_aux[0], axis=0), posinf=0, neginf=0, nan=0)))
        r2_per_fold.append(np.mean(r2_metric(y_test_aux, y_pred)))

        fold_no = fold_no + 1

    print('------------------------------------------------------------------------')
    print("Average MAE:", round(np.mean(mae_per_fold),3), "±", round(np.std(mae_per_fold),3))
    print("Average RMSE:", round(np.mean(rmse_per_fold),3), "±", round(np.std(rmse_per_fold),3))
    print("Average NE:", round(np.mean(normalized_error_per_fold),3), "±", round(np.std(normalized_error_per_fold),3))
    print("Avergae R2:", round(np.mean(r2_per_fold), 3), '±', round(np.std(r2_per_fold), 3))

    if serialize:
        # Write the predictions to a CSV file
        predictions_matrix.sort(key=lambda x: x[0])
        predictions_df = pd.DataFrame([x[1] for x in predictions_matrix], index = X.index).T
        predictions_df.to_csv('results/predictions_'+name+'.csv', encoding='utf-8', index=False)


def cross_validate_nn(dataset_opt, fix_uptakes, omics, use_deviations, use_flux_mask, max_trials, search_epochs, seed, name, batch_size, serialize):
    X, y, flux_mask = get_data(dataset_opt, fix_uptakes, omics, use_deviations, use_flux_mask)

    kfold = KFold(n_splits=len(X), shuffle=True)

    rmse_per_fold = []
    mae_per_fold = []
    normalized_error_per_fold = []
    r2_per_fold = []
    fold_no = 1
    if serialize:
        f = open('results/results_'+name+'.txt', 'w')
        predictions_matrix = []
    for train, test in kfold.split(X, y):
        tf.keras.backend.clear_session()

        # Basic Data Preprocessing
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X.values[train])
        y_train = sc_y.fit_transform(y.values[train])
        X_test = sc_X.transform(X.values[test])
        y_test = sc_y.transform(y.values[test])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        tuner = inner_cv(BayesianOptimization)(
        learning.HyperModel(y.shape[1], flux_mask),
        KFold(n_splits=5, random_state=seed, shuffle=True),
        objective=keras_tuner.Objective("val_loss", direction="min"),
        project_name="Model Optimization",
        directory='tuners/tuner_'+name+'_'+str(uuid.uuid1()),
        seed=seed,
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
            verbose=True,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        )

        # This is the list of the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]      
        
        model = tuner.hypermodel.build(best_hps)

        # Re-fit with external validation training data
        model.fit(X_train, y_train, batch_size=batch_size, epochs=search_epochs)
            
        # Inverse transform targets to extract metrics
        y_pred = model.predict(X_test)

        y_pred = sc_y.inverse_transform(y_pred)
        y_test_aux = sc_y.inverse_transform(y_test)

        # Generate generalization metrics
        rmse = np.sqrt(tf.keras.losses.MeanSquaredError()(y_test_aux,y_pred).numpy())
        mae = tf.keras.losses.MeanAbsoluteError()(y_test_aux,y_pred).numpy()
        ne = learning.NE_Loss()(y_test_aux,y_pred).numpy()
        r2 = np.mean(r2_metric(y_test_aux, y_pred))

        fold_message = f'Score for fold {fold_no}: RMSE of {rmse}; MAE of {mae}; Normalized error of {ne}'
        print(fold_message)

        if serialize:
            predictions_matrix.append((test[0], y_pred[0]))
            f.write(fold_message)
            f.write("\n")
            model_json = model.to_json()
            json_file = open("serialized/model_"+name+'-'+str(test[0])+".json", "w")
            json_file.write(model_json)
            json_file.write("\n")

        rmse_per_fold.append(rmse)
        mae_per_fold.append(mae)
        normalized_error_per_fold.append(ne)
        r2_per_fold.append(r2)

        fold_no = fold_no + 1

    output = ""

    output += "Average MAE:"+str(round(np.mean(mae_per_fold),3))+"±"+str(round(np.std(mae_per_fold),3))
    output += "\nAverage RMSE:"+str(round(np.mean(rmse_per_fold),3))+"±"+str(round(np.std(rmse_per_fold),3))
    output += "\nAverage NE:"+str(round(np.mean(normalized_error_per_fold),3))+"±"+str(round(np.std(normalized_error_per_fold),3))
    output += "\nAverage R²:"+str(round(np.mean(r2_per_fold),3))+"±"+str(round(np.std(r2_per_fold),3))

    print('------------------------------------------------------------------------')
    print(output)

    if serialize:
        # Write the final averaged results to a file
        f.write(output)
        f.close()
        
        # Write the predictions to a CSV file
        predictions_matrix.sort(key=lambda x: x[0])
        predictions_df = pd.DataFrame([x[1] for x in predictions_matrix], index = X.index).T
        predictions_df.to_csv('results/predictions_'+name+'.csv', encoding='utf-8', index=False)


def main():
    """
    Example commands:
        'python models.py --function external_validate'
        'python models.py --function cross_validate --model dt --dataset holm --omics proteomics transcriptomics --no-fix_uptakes --serialize'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="default") # dt, lr, rf, svm, xgb, nn
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--name', type=str, default="default")

    # Data specs               
    parser.add_argument('--dataset', type=str, default="ishii")                 
    parser.add_argument('--omics', nargs='+', default=['transcriptomics'])           
    parser.add_argument('--fix_uptakes', action=argparse.BooleanOptionalAction, default=True)        
    parser.add_argument('--use_deviations', action=argparse.BooleanOptionalAction, default=False)

    # NN exclusive options
    parser.add_argument('--device', type=str, default="CPU")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--memory_limit', type=int, default=3000)
    parser.add_argument('--eagerly', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max_trials', type=int, default=10)
    parser.add_argument('--search_epochs', type=int, default=10)
    parser.add_argument('--use_flux_mask', action=argparse.BooleanOptionalAction, default=False)

    # Execution Modes
    parser.add_argument('--serialize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--function', type=str, default="cross_validate") # cross_validate, external_validate

    opt = parser.parse_args()

    if opt.device == 'GPU':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=opt.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    if opt.function == 'cross_validate':
        if opt.model == 'nn':
            with tf.device('/'+opt.device+':0'):
                if opt.eagerly:
                    tf.config.run_functions_eagerly(run_eagerly=True)
                cross_validate_nn(opt.dataset,
                                  opt.fix_uptakes,
                                  opt.omics,
                                  opt.use_deviations,
                                  opt.use_flux_mask,
                                  opt.max_trials,
                                  opt.search_epochs,
                                  opt.seed,
                                  opt.name,
                                  opt.batch_size,
                                  opt.serialize)
        else:
            cross_validate(opt.model,
                           opt.dataset,
                           opt.fix_uptakes,
                           opt.omics,
                           opt.use_deviations,
                           opt.name,
                           opt.serialize)
            
    elif opt.function == 'external_validate':
        external_validate(opt.fix_uptakes,
                          opt.use_deviations,
                          opt.name,
                          opt.serialize,
                          opt.batch_size,
                          opt.max_trials,
                          opt.search_epochs)


if __name__ == "__main__":
    main()