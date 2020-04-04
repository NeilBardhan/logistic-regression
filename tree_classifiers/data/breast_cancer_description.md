## Description of breast cancer data

[Data Source Archive UCI ML](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

[UCI ML Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Attribute Information:

  1. `ID` number 
  2. `Diagnosis` 
      * `M` = malignant
      * `B` = benign 

Ten real-valued features are computed for each cell nucleus: 

* radius (mean of distances from center to points on the perimeter) 
* texture (standard deviation of gray-scale values) 
* perimeter 
* area 
* smoothness (local variation in radius lengths) 
* compactness (perimeter^2 / area - 1.0) 
* concavity (severity of concave portions of the contour) 
* concave points (number of concave portions of the contour) 
* symmetry 
* fractal dimension ("coastline approximation" - 1)

All feature values are recoded with four significant digits.

Missing attribute values: `None`

Class distribution: 

* `B` (benign) : 357
* `M` (malignant) : 212
