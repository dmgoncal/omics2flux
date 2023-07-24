configs = {
            "ishii":{
                    "proteomics" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/protein_nmol_per_gDW.csv',
                    "proteomics_std" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/protein_std_nmol_per_gDW.csv',
                    "transcriptomics" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/transcriptomics_nmol_per_gDW.csv',
                    "transcriptomics_std" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/transcriptomics_std_nmol_per_gDW.csv',
                    "fluxomics" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/ishii/fluxomics_mmol_per_gDW_per_h.csv',
                    "prior_fluxes" : ['R_EX_glc_e_','R_EX_o2_e_'],
                    "flux_mask" : {
                                    True : [1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
                                    False : [1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]
                                    }
                    },
            "holm":{
                    "proteomics" : None,
                    "proteomics_std" : None,
                    "transcriptomics" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/holm/transcriptome.csv',
                    "transcriptomics_std" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/holm/transcriptomics_std.csv',
                    "fluxomics" : 'https://raw.githubusercontent.com/cdanielmachado/transcript2flux/master/datasets/holm/fluxes_mmol_per_gDW_per_h.csv',
                    "prior_fluxes" : ['R_EX_glc_e_'],
                    "flux_mask" : {
                                    True : [1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,1,-1,1],
                                    False : [1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,1,-1,1]
                                    }
                    },

            "new_dataset":{
                    "proteomics" : None,
                    "proteomics_std" : None,
                    "transcriptomics" : None,
                    "transcriptomics_std" : None,
                    "fluxomics" : None,
                    "prior_fluxes" : [],
                    "flux_mask" : {
                                    True : None,
                                    False : None
                                    }
                    }
        }