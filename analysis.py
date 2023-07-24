import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from utils import mae_metric, rmse_metric, ne_metric, r2_metric, custom_sort

import warnings
warnings.filterwarnings("ignore")


epsilon=1e-9
path_to_files = "results/"

true_flux_ishii = pd.read_csv('https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/fluxomics_mmol_per_gDW_per_h.csv', index_col=0).drop(labels=['R_EX_glc_e_','R_EX_o2_e_'], axis=0).T
pfba_flux_ishii = pd.read_csv(path_to_files+'pFBA_ishii.csv', index_col=0).T

true_flux_holm = pd.read_csv('https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/holm/fluxes_mmol_per_gDW_per_h.csv', index_col=0).drop(labels=['R_EX_glc_e_'], axis=0)
true_flux_holm = true_flux_holm.loc[['R_GLCptspp', 'R_PGI', 'R_PFK', 'R_FBA', 'R_TPI', 'R_PGK', 'R_GAPD','R_ENO', 'R_PGM', 'R_PYK', 'R_PDH', 'R_G6PDH2r', 'R_PGL', 'R_GND','R_RPE', 'R_RPI', 'R_TKT1', 'R_TALA', 'R_TKT2', 'R_CS', 'R_ACONTb','R_ACONTa', 'R_ICDHyr', 'R_SUCOAS', 'R_AKGDH', 'R_SUCDi', 'R_FUM','R_MDH', 'R_PPC', 'R_ACKr', 'R_PTAr', 'R_EX_ac_e_','R_Ec_biomass_iAF1260_core_59p81M'],:].T
pfba_flux_holm = pd.read_csv(path_to_files+'pFBA_holm.csv', index_col=0).T

nn_flux = pd.read_csv(path_to_files+'predictions_dl_ishii_prot_fixed_deviations.csv').T
lr_flux = pd.read_csv(path_to_files+'predictions_lr_ishii_transprot_fixed_nodeviations.csv').T
svm_flux = pd.read_csv(path_to_files+'predictions_svm_ishii_prot_fixed_nodeviations.csv').T
dt_flux = pd.read_csv(path_to_files+'predictions_dt_ishii_trans_fixed_nodeviations.csv').T
rf_flux = pd.read_csv(path_to_files+'predictions_rf_ishii_trans_fixed_nodeviations.csv').T
xgb_flux = pd.read_csv(path_to_files+'predictions_xgb_ishii_trans_fixed_nodeviations.csv').T


def detailed_error_analysis(opt=1):
    """
    In depth error analysis (i.e. error by flux (column), error by setting (line), error for extracellular fluxes, error for intracellular fluxes)

    Args:
        opt (int): An option that dictates what models to consider (only two are implemented, see comments below)
    """
    if opt == 1:    # Consider the best models
        true_flux = true_flux_ishii
        pfba_flux = pfba_flux_ishii
        prediction_list = [pfba_flux, nn_flux, lr_flux, svm_flux, dt_flux, rf_flux, xgb_flux]
        label_list = ['pfba_flux / true', 'nn_flux / true', 'lr_flux / true', 'svm_flux / true', 'dt_flux / true', 'rf_flux / true', 'xgb_flux / true']
        
    elif opt == 2:  # Consider the 3 data inputs for RFs (intra_vs_extra_rf figure)
        true_flux = true_flux_ishii
        pfba_flux = pfba_flux_ishii
        prediction_list = [pd.read_csv(path_to_files+'predictions_rf_ishii_prot_fixed_nodeviations.csv').T,
                            pd.read_csv(path_to_files+'predictions_rf_ishii_trans_fixed_nodeviations.csv').T,
                            pd.read_csv(path_to_files+'predictions_rf_ishii_transprot_fixed_nodeviations.csv').T]
        label_list = ['RF (prot) / true', 'RF (trans) / true', 'RF (transprot) / true']

    for p in range(len(prediction_list)):
        print('\n############################## ' + label_list[p] + ' ##############################')

        print("\n[Error by SETTING (X)]")
        mae = mae_metric(true_flux.values, prediction_list[p].values)
        rmse = rmse_metric(true_flux.values, prediction_list[p].values)
        ne = ne_metric(true_flux.values, prediction_list[p].values)
        r2 = r2_metric(true_flux.values, prediction_list[p].values)
        df = pd.DataFrame([mae, rmse, ne, r2], columns = list(true_flux.index), index = ['MAE', 'RMSE', 'NE', 'R²'])
        print(df)

        print("\n[General Average Error]")
        print(df.mean(axis=1))

        print("\n[Error by FLUX (Y)]")
        mae = mae_metric(true_flux.values, prediction_list[p].values, axis = 0) # axis = 0 means average across lines (vertically = fluxes)
        rmse = rmse_metric(true_flux.values, prediction_list[p].values, axis = 0)
        ne = ne_metric(true_flux.values, prediction_list[p].values, axis = 0)
        r2 = r2_metric(true_flux.T.values, prediction_list[p].T.values)
        df = pd.DataFrame([mae, rmse, ne, r2], columns = pfba_flux.columns, index = ['MAE', 'RMSE', 'NE', 'R²'])
        print(df)

        print("\n[Average error by INTRA cellular flux]")
        mae_intra = mae_metric(true_flux.values[:,0:37], prediction_list[p].values[:,0:37])
        rmse_intra = rmse_metric(true_flux.values[:,0:37], prediction_list[p].values[:,0:37])
        ne_intra = ne_metric(true_flux.values[:,0:37], prediction_list[p].values[:,0:37])
        r2_intra = r2_metric(true_flux.values[:,0:37], prediction_list[p].values[:,0:37]) # evaluate intra section from each instance
        print('MAE intra:', '\t', np.mean(mae_intra), '±', np.std(mae_intra))
        print('RMSE intra:', '\t', np.mean(rmse_intra), '±', np.std(rmse_intra))
        print('NE intra:', '\t', np.mean(ne_intra), '±', np.std(ne_intra))
        print('R² intra:', '\t', np.mean(r2_intra), '±', np.std(r2_intra))

        print("\n[Average error by EXTRA cellular flux]")
        mae_extra = mae_metric(true_flux.values[:,37:-1], prediction_list[p].values[:,37:-1])
        rmse_extra = rmse_metric(true_flux.values[:,37:-1], prediction_list[p].values[:,37:-1])
        ne_extra = ne_metric(true_flux.values[:,37:-1], prediction_list[p].values[:,37:-1])
        r2_extra = r2_metric(true_flux.values[:,37:-1], prediction_list[p].values[:,37:-1]) # evaluate extra section from each instance
        print('MAE extra:', '\t', np.mean(mae_extra), '±', np.std(mae_extra))
        print('RMSE extra:', '\t', np.mean(rmse_extra), '±', np.std(rmse_extra))
        print('NE extra:', '\t', np.mean(ne_extra), '±', np.std(ne_extra))
        print('R² extra:', '\t', np.mean(r2_extra), '±', np.std(r2_extra))       


def calculate_metrics_overall(sep=','):
    """
    Wilcoxon Signed Rank Statistical analysis (all vs pFBA - for the ishii dataset only)
    """
    entries = os.listdir(path_to_files)
    filtered = list(filter(lambda file_name: file_name.startswith("predictions_") and "_nomask" not in file_name, entries))
    filtered.sort(key=custom_sort)

    res = []
    print('MAE ± SD, RMSE ± SD, NE ± SD, R² ± SD, p-value')
    for m in filtered + ['pfba_ishii', 'pfba_holm']:
        if 'holm' in m:
            true_flux = true_flux_holm
        else:
            true_flux = true_flux_ishii

        if m == 'pfba_ishii':
            current = pfba_flux_ishii
        elif m == 'pfba_holm':
            current = pfba_flux_holm
        else:
            current = pd.read_csv(path_to_files+m).T

        if current.shape[1] == 47:
            current = current.drop(current.iloc[:,[37,38]], axis=1)
        elif current.shape[1] == 34:   
            current = current.drop(current.iloc[:,[-3]], axis=1)

        mae = mae_metric(true_flux.values, current.values)
        rmse = rmse_metric(true_flux.values, current.values)
        ne = ne_metric(true_flux.values, current.values)
        r2 = r2_metric(true_flux.values, current.values)

        out_line = f'{m}\t{round(np.mean(mae),3)} ± {round(np.std(mae),3)}{sep}{round(np.mean(rmse),3)} ± {round(np.std(rmse),3)}{sep}{round(np.mean(ne),3)} ± {round(np.std(ne),3)}{sep}{round(np.mean(r2),3)} ± {round(np.std(r2),3)}'

        if '_holm' not in m and 'pfba' not in m:
            wil = wilcoxon((pfba_flux_ishii.values-current.values)+epsilon)
            print(out_line + sep + '{:.2e}'.format(np.median(wil.pvalue),3))
        else:
            print(out_line)


if __name__ == "__main__":
    detailed_error_analysis(opt=1)
    detailed_error_analysis(opt=2)
    calculate_metrics_overall()