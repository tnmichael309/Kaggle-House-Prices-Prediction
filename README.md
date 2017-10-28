# Kaggle-House-Prices-Prediction
House Prices Prediction: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

For the data preprocessing part, I referenced 
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard and https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

For the stacking model, I referenced
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard 

Also, I further extend the stacking model by supporting it with semi-supervised\unsupervised models as base models for meta-feature creations.

## Data Preprocessing
* Feature engineering
* Transform with RobustScaler
* Get dummies for categories variables (one-hot encodings)

## Model
* Stacking averaged model 
  * Base models for creating meta features: 
    * Supervised: ann, elasticnet, gradient boost, kernel ridge with out-of-fold predictions
    * Semi-supervised: knn 8, 16, 32 for distances to the k- nearest neighbors  + out-of-fold predictions
    * unsupervised: AffinityPropagation, mean_shift, k-means 8, 16, 32 for separating features into groups
  * meta model: lasso
* xgb: xg boost
* lgbm: light GBM

## Final Result:
* Best model: Stacking averaged model (all ml above used except ann) + xg boost + lightGBM
* Final Score on public leaderboard: rmse=0.11517 (10 %)

## Some Lessons:
* L1 regularization is also important, since it comes up with more sparse output than L2 regularization.
  * Also called built-in feature selection
* Stacking averaged model is commonly used to boost the final performance.
* Thourough cross-validation can provide more indication for the improvement on the test data, in which we don't have any correct answer.
  * With good cv, the cv scores go up with LB score.
  
