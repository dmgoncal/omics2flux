# ML2Flux

## About

This project aims to predict metabolic fluxes from omics data with machine learning models.

## File/Content Description

•	/plots -  This directory contains all graphical results, including figures that are referenced in the published article

•	/results - This directory contains a detailed set of results from all predictors, including predicted flux values and statistical significance test results against pFBA

•	/sbml - SBML model(s) used for pFBA simulations are stored in this folder

•	`analysis.py` - This script is responsible for the analytical post-processing of the results

•	`data.py` - Contains data sources and relevant configurations for data loading

•	`learning.py` - This script encapsulates the logic for implementing Neural Networks

•	`models.py` -  This script performs all tasks related to model fitting, cross-validation, and independent testing

•	`pfba.py` - Script that performs pFBA simulations

•	`plots.py` - Generates all the figures found in the /plots directory

•	requirements.txt -  Lists the packages required to replicate the development environment, e.g., using conda

•	`utils.py` - Contains miscellaneous auxiliary functions.

## Citation
This archive contains results, code and data utilized in the paper: "Predicting metabolic fluxes from omics data via machine learning: Moving from knowledge-driven towards data-driven approaches". Computational and Structural Biotechnology Journal (2023) | DOI: https://doi.org/10.1016/j.csbj.2023.10.002
