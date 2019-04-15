# Heart Disease Prediction

<p align = "center">
 +---------------------+ <br>
 |   Work in progress  | <br>
 +---------------------+ <br>
</p>

  * Logistic Regression
  * Decision Tree
  * Random Forest

## Data
The data is a modified version of the [UCI Heart Disease dataset](http://archive.ics.uci.edu/ml/datasets/heart+disease). My aim is to build a simple logistic regression classifier to predict the binary target. Then we will explore ways to fine tune that approach and build from there using decision trees and random forests.

From the UCI website, we get this 

The `goal` field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish **presence** (values 1,2,3,4) from **absence** (value 0). 

Attribute Information:

Only 14 attributes used: 
1. `age` : age in years 
2. `sex` : sex 
    * `1` : male
    * `0` : female 
3. `cp` : chest pain type 
    * `1` : typical angina 
    * `2` : atypical angina 
    * `3` : non-anginal pain 
    * `4` : asymptomatic
4. `trestbps` : resting blood pressure (in mm Hg on admission to the hospital) 
5. `chol` : serum cholestoral in mg/dl 
6. `fbs` : (fasting blood sugar > 120 mg/dl) 
    * `1` : true
    * `0` : false
7. `restecg` : resting electrocardiographic results 
    * `0` : normal 
    * `1` : having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
    * `2` : showing probable or definite left ventricular hypertrophy by Estes' criteria 
8. `thalach` : maximum heart rate achieved 
9. `exang` : exercise induced angina
    * `1` : yes
    * `0` : no
10. `oldpeak` : ST depression induced by exercise relative to rest 
11. `slope` : the slope of the peak exercise ST segment 
    * `1` : upsloping 
    * `2` : flat 
    * `3` : downsloping
12. `ca` : number of major vessels (0-3) colored by flourosopy 
13. `thal` : 
    * `3` : normal
    * `6` : fixed defect
    * `7` : reversable defect
14. `num` (**target**) : diagnosis of heart disease (angiographic disease status) 
    * `0` : < 50% diameter narrowing 
    * `1` : > 50% diameter narrowing
