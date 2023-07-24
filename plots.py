import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils import replace_strings_with_dictionary_values


path_to_files = "results/"
epsilon=1e-9
overall = pd.read_csv(path_to_files+'overall_results.csv')


def plot_all_fluxes_holm():

    true_flux = pd.read_csv('https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/holm/fluxes_mmol_per_gDW_per_h.csv', index_col=0).drop(labels=['R_EX_glc_e_'], axis=0)
    true_flux = true_flux.loc[['R_GLCptspp', 'R_PGI', 'R_PFK', 'R_FBA', 'R_TPI', 'R_PGK', 'R_GAPD','R_ENO', 'R_PGM', 'R_PYK', 'R_PDH', 'R_G6PDH2r', 'R_PGL', 'R_GND','R_RPE', 'R_RPI', 'R_TKT1', 'R_TALA', 'R_TKT2', 'R_CS', 'R_ACONTb','R_ACONTa', 'R_ICDHyr', 'R_SUCOAS', 'R_AKGDH', 'R_SUCDi', 'R_FUM','R_MDH', 'R_PPC', 'R_ACKr', 'R_PTAr', 'R_EX_ac_e_','R_Ec_biomass_iAF1260_core_59p81M'],:]
    pfba_flux = pd.read_csv(path_to_files+'pFBA_holm.csv', index_col=0)
    rf_flux = pd.read_csv(path_to_files+'predictions_holm_rf_trans_fixed_nodeviations.csv')


    mydict = {'EX_co2_e_':'EX_CO2',
              'EX_etoh_e_':'EX_Ethanol',
              'EX_ac_e_':'EX_Acetate',
              'EX_lac_D_e_':'EX_Lactate',
              'EX_succ_e_':'EX_Succinate',
              'EX_pyr_e_':'EX_Pyruvate',
              'EX_for_e_':'EX_Formate'}
    
    metabolite_names = replace_strings_with_dictionary_values(list(pfba_flux.index), mydict)
    
    params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large'}
    
    pylab.rcParams.update(params)

    n_cols = 6
    n_lins = 6
    for s in range(0,len(list(true_flux.columns))): # experimental setting (dilution, knockout)
        fig, axs = plt.subplots(n_lins, n_cols, figsize=(18, 25))
        for m in range(0,33): # metabolites
            axs[m//n_cols, m%n_cols].bar(['True', 'pFBA', 'RF'], 
                        [true_flux.to_numpy()[m,s],
                            pfba_flux.to_numpy()[m,s],
                            rf_flux.to_numpy()[m,s]],
                        color = ['#FFB6C1','#93b2c7','#93b2c7'],
                        bottom=0, linewidth=0.2, edgecolor='gray', alpha = 0.95, hatch = ['\\\\','',''])
            
            axs[m//n_cols, m%n_cols].set_title(('Growth' if 'biomass' in metabolite_names[m] else metabolite_names[m]))
            axs[m//n_cols, m%n_cols].set_xticklabels(axs[m//n_cols, m%n_cols].get_xticklabels(), rotation=90)

            if m//n_cols != 5:
                axs[m//n_cols, m%n_cols].set_xticks([])

            t = axs[m//n_cols, m%n_cols].yaxis.get_offset_text()
            t.set_x(0.01)
        axs[-1,-1].set_visible(False)
        axs[-1,-2].set_visible(False)
        axs[-1,-3].set_visible(False)

        fig.tight_layout(pad=1)
        fig.text(0.5, 0.05, 'Approach', ha='center', fontsize='x-large')
        fig.text(0.05, 0.5, 'Flux', va='center', rotation='vertical', fontsize='x-large')

        plt.subplots_adjust(bottom=0.09,left=0.09)
        plt.savefig("plots/holm_"+list(true_flux.columns)[s]+".pdf", format="pdf", bbox_inches="tight")
        plt.clf()


def plot_all_fluxes():
    """
    Plot the detailed view of the predictions from the best ML models and pFBA for all available fluxes.
    """
    # Best models
    true_flux = pd.read_csv('https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/fluxomics_mmol_per_gDW_per_h.csv', index_col=0).drop(labels=['R_EX_glc_e_','R_EX_o2_e_'], axis=0)
    pfba_flux = pd.read_csv(path_to_files+'pFBA_ishii.csv', index_col=0)
    nn_flux = pd.read_csv(path_to_files+'predictions_dl_ishii_prot_fixed_deviations.csv')
    lr_flux = pd.read_csv(path_to_files+'predictions_lr_ishii_transprot_fixed_nodeviations.csv')
    svm_flux = pd.read_csv(path_to_files+'predictions_svm_ishii_prot_fixed_nodeviations.csv')
    dt_flux = pd.read_csv(path_to_files+'predictions_dt_ishii_trans_fixed_nodeviations.csv')
    rf_flux = pd.read_csv(path_to_files+'predictions_rf_ishii_trans_fixed_nodeviations.csv')
    xgb_flux = pd.read_csv(path_to_files+'predictions_xgb_ishii_trans_fixed_nodeviations.csv')

    mydict = {'EX_co2_e_':'EX_CO2',
              'EX_etoh_e_':'EX_Ethanol',
              'EX_ac_e_':'EX_Acetate',
              'EX_lac_D_e_':'EX_Lactate',
              'EX_succ_e_':'EX_Succinate',
              'EX_pyr_e_':'EX_Pyruvate',
              'EX_for_e_':'EX_Formate'}
    
    metabolite_names = replace_strings_with_dictionary_values(list(pfba_flux.index), mydict)

    predictions = []
    original_cols = []
    first_flag = True
    for file in os.listdir(path_to_files):
        if file.startswith("predictions_"):
            df = pd.read_csv(path_to_files+file)
            if df.shape[0] == 47:
                df = df.drop(labels=[37,38], axis=0)
            predictions.append(df.to_numpy())
            if first_flag:
                original_cols = df.columns
    
    params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large'}
    
    pylab.rcParams.update(params)

    predictions=np.array(predictions)
    n_cols = 5
    n_lins = 9
    for s in range(0,len(original_cols)): # experimental setting (dilution, knockout)
        fig, axs = plt.subplots(n_lins, n_cols, figsize=(18, 25))
        for m in range(0,45): # metabolites
            axs[m//n_cols, m%n_cols].bar(['True', 'pFBA', 'LR', 'SVM', 'DT', 'RF', 'XGB', 'NN'], 
                        [true_flux.to_numpy()[m,s],
                            pfba_flux.to_numpy()[m,s],
                            lr_flux.to_numpy()[m,s], 
                            svm_flux.to_numpy()[m,s],
                            dt_flux.to_numpy()[m,s], 
                            rf_flux.to_numpy()[m,s], 
                            xgb_flux.to_numpy()[m,s], 
                        nn_flux.to_numpy()[m,s]],
                        color = ['#FFB6C1','#93b2c7','#93b2c7','#93b2c7','#93b2c7','#93b2c7','#93b2c7','#93b2c7'],
                        bottom=0, linewidth=0.2, edgecolor='gray', alpha = 0.95, hatch = ['\\\\','','','','','','',''])
            
            axs[m//n_cols, m%n_cols].set_title(('Growth' if 'biomass' in metabolite_names[m] else metabolite_names[m]))
            axs[m//n_cols, m%n_cols].set_xticklabels(axs[m//n_cols, m%n_cols].get_xticklabels(), rotation=90)

            if m//n_cols != 8:
                axs[m//n_cols, m%n_cols].set_xticks([])

            t = axs[m//n_cols, m%n_cols].yaxis.get_offset_text()
            t.set_x(0.01)

        fig.tight_layout(pad=1)
        fig.text(0.5, 0.05, 'Approach', ha='center', fontsize='x-large')
        fig.text(0.05, 0.5, 'Flux', va='center', rotation='vertical', fontsize='x-large')

        plt.subplots_adjust(bottom=0.09,left=0.09)
        plt.savefig("plots/"+original_cols[s]+".pdf", format="pdf", bbox_inches="tight")
        plt.clf()


def plot_best_models():
    """
    Plot the best model comparison bar plot.
    """
    selected = overall.iloc[[41,49,17,65,0,33,72]]

    labels = selected.loc[:,'Model']
    mae_means = selected.loc[:,'MAE']
    mse_means = selected.loc[:,'RMSE']
    ne_means = selected.loc[:,'NE']
    r2_means = selected.loc[:,'R2']

    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars

    fig, ax = plt.subplots(figsize=(20,10))
    rects1 = ax.bar(x - 1.5*width, mae_means, width, label='MAE', align='center', zorder=3, color='#9BC1E2', linewidth=0.2, edgecolor='gray', alpha=0.95)
    rects2 = ax.bar(x - width/2, mse_means, width, label='RMSE', align='center', zorder=3, color='#FDBE88', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='/')
    rects3 = ax.bar(x + width/2, ne_means, width, label='NE', align='center', zorder=3, color='#9ED8B4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='.')
    rects4 = ax.bar(x + 1.5*width, r2_means, width, label='R²', align='center', zorder=3, color='#D3B9E4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='x')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25)
    ax.legend(prop={'size': 25}, loc='upper right', bbox_to_anchor=(1.1, 1.15))

    fig.tight_layout()
    plt.grid(True,axis='y', zorder=0, linestyle='--')
    plt.savefig("plots/best_models.pdf", format="pdf", bbox_inches="tight")


def plot_std_analysis():
    """
    Plot the standard deviation analysis plot.
    """
    selected = overall.iloc[[36,40,44,37,41,45]]

    labels = selected.loc[:,'Omics']
    mae_means = selected.loc[:,'MAE']
    mse_means = selected.loc[:,'RMSE']
    ne_means = selected.loc[:,'NE']
    r2_means = selected.loc[:,'R2']

    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars

    fig, ax = plt.subplots(figsize=(20,10))
    rects1 = ax.bar(x - 1.5*width, mae_means, width, label='MAE', align='center', zorder=3, color='#9BC1E2', linewidth=0.2, edgecolor='gray', alpha=0.95)
    rects2 = ax.bar(x - width/2, mse_means, width, label='RMSE', align='center', zorder=3, color='#FDBE88', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='/')
    rects3 = ax.bar(x + width/2, ne_means, width, label='NE', align='center', zorder=3, color='#9ED8B4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='.')
    rects4 = ax.bar(x + 1.5*width, r2_means, width, label='R²', align='center', zorder=3, color='#D3B9E4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='x')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25)
    ax.legend(prop={'size': 25}, loc='upper right', bbox_to_anchor=(1.1, 1.15))

    fig.tight_layout()
    # Get the x-axis limits
    x_min, x_max = plt.xlim()

    # Calculate the middle position
    middle = (x_min + x_max) / 2

    # Plot the dashed line at the middle position
    plt.axvline(x=middle, color='black', linestyle='-')
    plt.text(0.8, 1.03, "With", color='black', fontsize=25)
    plt.text(3.89, 1.03, "Without", color='black', fontsize=25)
    plt.grid(True,axis='y', zorder=0, linestyle='--')
    plt.savefig("plots/deviations_analysis.pdf", format="pdf", bbox_inches="tight")


def plot_intra_vs_extra_rf():
    """
    Plot intra vs extracellular fluxes error views for Random Forest model
    """  
    selected = overall.iloc[[36,40,44,37,41,45]]

    labels = selected.loc[:,'Omics']

    mae_means = [0.654,0.571,0.578,  0.207,0.204,0.201]
    mse_means = [0.845,0.733,0.748,  0.498,0.508,0.498]
    ne_means = [0.314,0.246,0.257,   0.858,0.774,0.789]
    r2_means = [0.980,0.976,0.977,   0.999,0.999,0.999]

    x = np.arange(len(labels))  # the label locations
    width = 0.20                # the width of the bars

    fig, ax = plt.subplots(figsize=(20,10))
    rects1 = ax.bar(x - 1.5*width, mae_means, width, label='MAE', align='center', zorder=3, color='#9BC1E2', linewidth=0.2, edgecolor='gray')
    rects2 = ax.bar(x - width/2, mse_means, width, label='RMSE', align='center', zorder=3, color='#FDBE88', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='/')
    rects3 = ax.bar(x + width/2, ne_means, width, label='NE', align='center', zorder=3, color='#9ED8B4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='.')
    rects4 = ax.bar(x + 1.5*width, r2_means, width, label='R²', align='center', zorder=3, color='#D3B9E4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='x')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)  # most of the data
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25)
    ax.legend(prop={'size': 25}, loc='upper right', bbox_to_anchor=(1.18, 1.115))

    fig.tight_layout()
    # Get the x-axis limits
    x_min, x_max = plt.xlim()

    # Calculate the middle position
    middle = (x_min + x_max) / 2

    # Plot the dashed line at the middle position
    plt.axvline(x=middle, color='black', linestyle='-')
    plt.text(0.8, 1.21, "Intra", color='black', fontsize=25)
    plt.text(3.86, 1.21, "Extra", color='black', fontsize=25)
    plt.grid(True,axis='y', zorder=0, linestyle='--')
    
    plt.savefig("plots/intravsextra_rf.pdf", format="pdf", bbox_inches="tight")


def plot_intra_vs_extra_pfba():
    """
    Plot intra vs extracellular fluxes error views for pFBA
    """
    labels = ['pFBA','pFBA']

    mae_means = [0.735,0.554]
    mse_means = [1.016,1.171]
    ne_means = [0.341,0.875]
    r2_means = [0.861,0.886]

    x = np.arange(len(labels))  # the label locations
    width = 0.20                # the width of the bars

    fig, ax = plt.subplots(figsize=(8,10))
    rects1 = ax.bar(x - 1.5*width, mae_means, width, label='MAE', align='center', zorder=3, color='#9BC1E2', linewidth=0.2, edgecolor='gray')
    rects2 = ax.bar(x - width/2, mse_means, width, label='RMSE', align='center', zorder=3, color='#FDBE88', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='/')
    rects3 = ax.bar(x + width/2, ne_means, width, label='NE', align='center', zorder=3, color='#9ED8B4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='.')
    rects4 = ax.bar(x + 1.5*width, r2_means, width, label='R²', align='center', zorder=3, color='#D3B9E4', linewidth=0.2, edgecolor='gray', alpha=0.95, hatch='x')

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(0, 1.2)  # most of the data
    ax.set_xlim(-0.7, 1.7)  # most of the data

    ax.xaxis.tick_bottom()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel(ax.get_xlabel(), fontsize=25)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    fig.tight_layout()
    # Get the x-axis limits
    x_min, x_max = plt.xlim()

    # Calculate the middle position
    middle = (x_min + x_max) / 2

    # Plot the dashed line at the middle position
    ax.axvline(x=middle, color='black', linestyle='-')
    plt.text(-0.25, 1.21, "Intra", color='black', fontsize=25)
    plt.text(0.95, 1.21, "Extra", color='black', fontsize=25)
    ax.grid(True,axis='y', zorder=0, linestyle='--')

    plt.savefig("plots/intravsextra_pfba.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":    
    plot_all_fluxes()
    plot_all_fluxes_holm()
    plot_best_models()
    plot_std_analysis()
    plot_intra_vs_extra_rf()
    plot_intra_vs_extra_pfba()