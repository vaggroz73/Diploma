# Diploma Thesis

This study proposes a longitudinal and explainable framework for predicting year-over-year changes in football player overall ratings. The framework integrates historical match statistics, advanced football indicators, player characteristics, and FIFA attributes from the five major European leagues. The final dataset contains 5,250 player-season observations and 95 features. The study compares conventional machine learning models, including Linear Regression, Ridge Regression, Random Forest, and XGBoost, with recurrent deep learning models based on Long Short-Term Memory and Gated Recurrent Unit architectures. A chronological data split preserves the temporal structure of the task and prevents information leakage. SHAP analysis identifies the factors that most strongly influence predicted rating changes. The novelty of this work lies in jointly modelling long-term player rating development and provides explainable insights into the prediction process. The proposed framework moves beyond short-term performance estimation and market value prediction and offers an interpretable approach that can support talent identification and player monitoring.

merg.py: This python file implements the full pipeline to merge the three heterogenous datasets (FIFA, Understat, FBref). Due to different players names across the datasets, RegEx and fuzzy mathing (with
         rappidfuzz library) lead to the creation of a unified final dataset.

corr.py: This python file helps with the creation of correlation matrixes in order to reveal the relations between different stats of football players and overall performance.

LSTM.py and GRU.py iclude the full pipeline for Neural Networks, in order to forecast the future performance. From Data Preprocessing and Feature Engineering to chronological split ensuring the prevention
of data leakage. After the training and evaluation on Evaluation set, monitoring the training phase helps to include Hyperparemeter Tuning. The final evaluation with regression metrics (MAE, RMSE, R-sqaured)
For interpretation, Shap Analysis condacted for global and local explanations (Feature Importance, Individual waterfall plots, heatmaps for feature importance of each timestep).

gui_2.py. The streamlit application implementing a friendly user interface for final results visualization comparing the Neural Networks perfomances and individual players results.



