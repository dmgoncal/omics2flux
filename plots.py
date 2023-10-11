import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from utils import replace_strings_with_dictionary_values
from utils import mae_metric, rmse_metric, ne_metric, r2_metric, get_data, adapt_val_to_train


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
        axs[4, -1].set_xticks(['True','pFBA','RF'])
        axs[4, -2].set_xticks(['True','pFBA','RF'])
        axs[4, -3].set_xticks(['True','pFBA','RF'])
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


def plot_flux_error_histogram():
    '''
    Plot error histograms by metabolic flux.
    '''
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
              'EX_for_e_':'EX_Formate',
              'Ec_biomass_iAF1260_core_59p81M' : 'Growth'}
    
    metabolite_names = replace_strings_with_dictionary_values(list(pfba_flux.index), mydict)

    pfba_mae = mae_metric(true_flux.values,pfba_flux.values)
    pfba_rmse = rmse_metric(true_flux.values,pfba_flux.values)
    pfba_ne = ne_metric(true_flux.values,pfba_flux.values)
    pfba_r2 = r2_metric(true_flux.values,pfba_flux.values)

    pfba_metrics = {'MAE' : pfba_mae, 'RMSE' : pfba_rmse, 'NE' : pfba_ne, 'R²' : pfba_r2}

    for model in [(rf_flux,'RF'), (nn_flux,'DL'), (lr_flux,'LR'), (svm_flux,'SVM'), (dt_flux,'DT'), (xgb_flux,'XGB')]:
        if model[0].shape[0] == 47:
            model[0] = model[0].drop(labels=[37,38], axis=0)

        model_mae = mae_metric(true_flux.values,model[0].values)
        model_rmse = rmse_metric(true_flux.values,model[0].values)
        model_ne = ne_metric(true_flux.values,model[0].values)
        model_r2 = r2_metric(true_flux.values,model[0].values)

        metrics = [(model_ne,'NE'),(model_mae,'MAE'),(model_rmse,'RMSE'),(model_r2,'R²')] 
        for metric,name in metrics:
            df = pd.DataFrame([metabolite_names, list(pfba_metrics[name]), list(metric)], index=['Fluxes', 'pfba', name]).T
            
            plt.figure(figsize=(11, 7)) 
            ax = sns.barplot(data=df, x="Fluxes", y="pfba", color = '#FDBE88', alpha=0.9, linewidth=0.5, edgecolor='gray', width=0.9)
            ax = sns.barplot(data=df, x="Fluxes", y=name, color = '#7cacd6', alpha=0.7, width=0.9, linewidth=0.5, edgecolor='gray', hatch='//')
            plt.xticks(rotation=90)
            plt.tight_layout()
            legend_elements = [mpatches.Patch(facecolor='#FDBE88', edgecolor='gray', label='pFBA'),
                                mpatches.Patch(facecolor='#7cacd6', edgecolor='gray', hatch='//', label=model[1])]
            plt.legend(handles=legend_elements, loc='upper left')


            '''For RF's NE only - cut y axis'''
            # f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 4],'hspace':0.05},figsize=(11, 7))
            # sns.barplot(data=df, x="Fluxes", y="pfba", color = '#FDBE88', alpha=0.9, linewidth=0.5, edgecolor='gray', width=0.9, ax=ax_top)
            # sns.barplot(data=df, x="Fluxes", y="pfba", color = '#FDBE88', alpha=0.9, linewidth=0.5, edgecolor='gray', width=0.9, ax=ax_bottom)
            # sns.barplot(data=df, x="Fluxes", y=name, color = '#7cacd6', alpha=0.7, width=0.9, linewidth=0.5, edgecolor='gray', hatch='//', ax=ax_top)
            # sns.barplot(data=df, x="Fluxes", y=name, color = '#7cacd6', alpha=0.7, width=0.9, linewidth=0.5, edgecolor='gray', hatch='//', ax=ax_bottom)

            # ax_top.set_ylim(bottom=25)   # those limits are fake
            # ax_bottom.set_ylim(0,8)

            # ax_top.spines[['bottom']].set_visible(False)
            # ax_bottom.spines[['top']].set_visible(False)

            # ax = ax_top
            # d = .003
            # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            # ax.plot((-d, +d), (-d, +d), **kwargs)
            # ax.plot((1 - d, 1 + d), (-d,+d), **kwargs)

            # ax2 = ax_bottom
            # kwargs.update(transform=ax2.transAxes)
            # ax2.plot((-d, +d), (1.005 - d, 1 + d), **kwargs)
            # ax2.plot((1 - d, 1 + d), (1.005-d, 1+d), **kwargs)
            # legend_elements = [mpatches.Patch(facecolor='#FDBE88', edgecolor='gray', label='pFBA'),
            #                     mpatches.Patch(facecolor='#7cacd6', edgecolor='gray', hatch='//', label=model[1])]
            # ax.legend(handles=legend_elements, loc='upper left')
            # ax.tick_params(bottom = False)
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            # ax2.set_xticks(ticks=range(len(metabolite_names)),labels=metabolite_names, rotation='vertical')

            plt.savefig("plots/error_histogram_"+model[1]+"_"+name+".pdf", format="pdf", bbox_inches='tight')
            exit()


def plot_learning_curves():
    '''
    Plot the learning curves for train (Ishii) and test (Holm) at incrementing training size.
    '''
    X_ishii_original, y_ishii_original, _ = get_data('ishii', True, ['transcriptomics'], False)
    X_holm, y_holm, _ = get_data('holm', True, ['transcriptomics'], False)

    X_holm = adapt_val_to_train(X_ishii_original,X_holm)
    y_holm = adapt_val_to_train(y_ishii_original,y_holm)
    
    X_ishii_mod = X_ishii_original[X_holm.columns].values
    y_ishii_mod = y_ishii_original[y_holm.columns].values
    X_holm = X_holm.values
    y_holm = y_holm.values

    X = X_ishii_original.values
    y = y_ishii_original.values

    n_runs = 10
    data_indices = list(range(len(X)))
    loss_ishii = []
    model = RandomForestRegressor()
    for i in range(1, len(X)-1):
        
        loss_cv = 0.0
        for runs in range(n_runs):
            training_indices = random.sample(data_indices, i)
            
            testing_indices = [idx for idx in data_indices if idx not in training_indices]
            
            X_train = X[training_indices]
            y_train = y[training_indices]
            X_test = X[testing_indices]
            y_test = y[testing_indices]
            
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            y_train = sc_y.fit_transform(y_train)
            X_test = sc_X.transform(X_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = sc_y.inverse_transform(y_pred)
            loss_cv += np.mean(ne_metric(y_pred, y_test))
        
        loss_cv /= n_runs
        loss_ishii.append(loss_cv)
    loss_ishii = np.array(loss_ishii)
    sns.lineplot(x=range(1,len(loss_ishii)+1), y=loss_ishii, label='Ishii')
    plt.xlabel("N. of training instances")
    plt.ylabel("NE Loss")

    loss_holm = []
    data_indices = list(range(len(X_ishii_mod)))
    for i in range(1, len(X_ishii_mod)-1):
        
        loss_cv = 0.0
        for runs in range(n_runs):
            training_indices = random.sample(data_indices, i)
            
            X_train = X_ishii_mod[training_indices]
            y_train = y_ishii_mod[training_indices]
            X_test = X_holm
            y_test = y_holm

            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            y_train = sc_y.fit_transform(y_train)
            X_test = sc_X.transform(X_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = sc_y.inverse_transform(y_pred)
            loss_cv += np.mean(ne_metric(y_pred, y_test))
        
        loss_cv /= n_runs
        loss_holm.append(loss_cv)

    sns.lineplot(x=range(1,len(loss_holm)+1), y=loss_holm,label='Holm')
    plt.legend()
    plt.savefig("plots/train_vs_test_loss.pdf", format="pdf")


if __name__ == "__main__":    
    plot_all_fluxes()
    plot_all_fluxes_holm()
    plot_best_models()
    plot_std_analysis()
    plot_intra_vs_extra_rf()
    plot_intra_vs_extra_pfba()
    plot_flux_error_histogram()
    plot_learning_curves()