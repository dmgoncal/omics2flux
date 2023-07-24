import cobra
import logging
import argparse
import numpy as np
import pandas as pd
from utils import get_data, mae_metric, rmse_metric, ne_metric, r2_metric


logging.getLogger("cobra").setLevel(logging.ERROR)

model = cobra.io.read_sbml_model('./sbml/iAF1260.xml')

dict_of_settings = {'ishii' : {
                        'settings': [
                            {'name': 'REF', 'dilution': 0.2, 'glc': -2.93, 'o2': -5.79},           
                            {'name': 'WT_0.1h-1', 'dilution': 0.1, 'glc': -1.34, 'o2': -2.2},       
                            {'name': 'WT_0.4h-1', 'dilution': 0.4, 'glc': -5.05, 'o2': -9.57},      
                            {'name': 'WT_0.5h-1', 'dilution': 0.5, 'glc': -6.63, 'o2': -13.05},     
                            {'name': 'WT_0.7h-1', 'dilution': 0.7, 'glc': -13.34, 'o2': -14.87},    
                            {'name': 'b0756', 'dilution': 0.2, 'glc': -3.04, 'o2': -15.53},        
                            {'name': 'b2388', 'dilution': 0.2, 'glc': -3.14, 'o2': -5.44},         
                            {'name': 'b0688', 'dilution': 0.2, 'glc': -3.06, 'o2': -5.79},         
                            {'name': 'b4025', 'dilution': 0.2, 'glc': -2.73, 'o2': -4.45},         
                            {'name': 'b3916', 'dilution': 0.2, 'glc': -2.72, 'o2': -9.56},         
                            {'name': 'b1723', 'dilution': 0.2, 'glc': -2.84, 'o2': -5.83},         
                            {'name': 'b4232', 'dilution': 0.2, 'glc': -2.7, 'o2': -6.0},           
                            {'name': 'b2097', 'dilution': 0.2, 'glc': -2.76, 'o2': -8.08},         
                            {'name': 'b0118', 'dilution': 0.2, 'glc': -2.99, 'o2': -6.79},         
                            {'name': 'b0755', 'dilution': 0.2, 'glc': -2.7, 'o2': -6.71},          
                            {'name': 'b4395', 'dilution': 0.2, 'glc': -3.33, 'o2': -5.05},         
                            {'name': 'b1854', 'dilution': 0.2, 'glc': -2.82, 'o2': -7.17},         
                            {'name': 'b1676', 'dilution': 0.2, 'glc': -2.8, 'o2': -6.06},          
                            {'name': 'b1702', 'dilution': 0.2, 'glc': -3.0, 'o2': -9.95},          
                            {'name': 'b1852', 'dilution': 0.2, 'glc': -3.18, 'o2': -8.76},         
                            {'name': 'b0767', 'dilution': 0.2, 'glc': -3.36, 'o2': -11.87},        
                            {'name': 'b2029', 'dilution': 0.2, 'glc': -2.92, 'o2': -6.01},         
                            {'name': 'b3386', 'dilution': 0.2, 'glc': -3.63, 'o2': -8.04},         
                            {'name': 'b2914', 'dilution': 0.2, 'glc': -3.18, 'o2': -3.02},         
                            {'name': 'b4090', 'dilution': 0.2, 'glc': -4.15, 'o2': -15.52},        
                            {'name': 'b2935', 'dilution': 0.2, 'glc': -4.49, 'o2': -9.0},          
                            {'name': 'b2465', 'dilution': 0.2, 'glc': -2.97, 'o2': -12.21},        
                            {'name': 'b2464', 'dilution': 0.2, 'glc': -2.8, 'o2': -3.4},           
                            {'name': 'b0008', 'dilution': 0.2, 'glc': -2.94, 'o2': -10.99}],       
                        'columns' : ['GLCptspp', 'PGI', 'PFK', 'FBA', 'TPI', 'PGK', 'GAPD', 'ENO', 'PGM', 'PYK', 'PDH', 'G6PDH2r', 'PGL', 'GND', 'RPE', 'RPI', 'TKT1', 'TALA', 'TKT2', 'CS', 'ACONTb', 'ACONTa', 'ICDHyr', 'SUCOAS', 'AKGDH', 'SUCDi', 'FUM', 'MDH', 'PPC', 'ME2', 'ICL', 'MALS', 'ACKr', 'PTAr', 'LDH_D', 'ACALD', 'ALCD2x', 'EX_co2_e_', 'EX_etoh_e_', 'EX_ac_e_', 'EX_lac_D_e_', 'EX_succ_e_', 'EX_pyr_e_', 'EX_for_e_', 'Ec_biomass_iAF1260_core_59p81M']
                        },

                    'holm' : {
                        'settings' : [
                            {'name': 'REF', 'glc': -9.2},
                            {'name': 'KNOX', 'glc': -11.7},
                            {'name': 'ATP', 'glc': -15.6}],
                        'columns' : ['GLCptspp', 'PGI', 'PFK', 'FBA', 'TPI', 'PGK', 'GAPD', 'ENO', 'PGM', 'PYK', 'PDH', 'G6PDH2r', 'PGL', 'GND', 'RPE', 'RPI', 'TKT1', 'TALA', 'TKT2', 'CS', 'ACONTb', 'ACONTa', 'ICDHyr', 'SUCOAS', 'AKGDH', 'SUCDi', 'FUM', 'MDH', 'PPC','ACKr', 'PTAr', 'EX_ac_e_', 'Ec_biomass_iAF1260_core_59p81M']
                        }
                    }


def pfba(data, serialize):
    predictions = []
    for settings in dict_of_settings[data]['settings']:
        with model:
            if data == 'ishii':
                if settings['name'] not in ['REF', 'WT_0.1h-1', 'WT_0.4h-1', 'WT_0.5h-1', 'WT_0.7h-1']:
                    getattr(model.genes,(settings['name'])).knock_out()

            glc_uptake_reaction = model.reactions.get_by_id("EX_glc_e_")
            glc_uptake_reaction.bounds = (settings['glc'], settings['glc'])
            if data == 'ishii':
                o2_uptake_reaction = model.reactions.get_by_id("EX_o2_e_")
                o2_uptake_reaction.bounds = (settings['o2'], settings['o2'])

            pfba_solution = cobra.flux_analysis.pfba(model)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                predictions.append(pfba_solution.fluxes[dict_of_settings[data]['columns']])

    fluxes_real = get_data(data, False, ['transcriptomics'], False, False)[1]
    fluxes_real = fluxes_real.loc[:,['R_'+c for c in dict_of_settings[data]['columns']]]

    result = pd.concat(predictions, axis=1)
    result.columns = [el['name'] for el in dict_of_settings[data]['settings']]

    if serialize:
        result.to_excel("./results/pFBA_"+data+".xlsx")
        result.to_csv("./results/pFBA_"+data+".csv")

    fluxes_predicted = result.T.to_numpy()
    fluxes_real = fluxes_real.to_numpy()

    print('[Results for '+data+' dataset]')
    print("Average MAE:", round(np.mean(mae_metric(fluxes_real,fluxes_predicted)), 3), "±", round(np.std(mae_metric(fluxes_real,fluxes_predicted)), 3))
    print("Average RMSE:", round(np.mean(rmse_metric(fluxes_real,fluxes_predicted)), 3), "±", round(np.std(rmse_metric(fluxes_real,fluxes_predicted)), 3))
    print("Average NE:", round(np.mean(ne_metric(fluxes_real, fluxes_predicted)), 3), "±", round(np.std(ne_metric(fluxes_real, fluxes_predicted)), 3))
    print("Avergae R2:", round(np.mean(r2_metric(fluxes_real, fluxes_predicted)), 3), '±', round(np.std(r2_metric(fluxes_real, fluxes_predicted)), 3))


def main():
    # Example command: 'python pfba.py --data ishii --serialize'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="ishii") # 'ishii' or 'holm'
    parser.add_argument('--serialize', action=argparse.BooleanOptionalAction, default=False)

    opt = parser.parse_args()

    pfba(opt.data, opt.serialize)


if __name__ == "__main__":
    main()