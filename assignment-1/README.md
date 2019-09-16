### Assignment 1
# Classification and Clustering using Weka Explorer

## Description
* Find dataset in UCI machine learning, Kaggle, etc, to be processed.
* Do **classification** using the following methods (supervised learning):
  * Naive Bayes
  * K-Nearest Neighbor (KNN)
  * Support Vector Machine (SVM)  
* Do **clustering** using the following methods (unsupervised learning):
  * K-Means
  
## Tools 
* Weka Explorer 3.8.2 

## Dataset Description
* Title : 
    **Forest Covertype data**
* Description : 
    **Forest Cover Type (FC) data** contains **tree observations** from four wilderness areas located in the Roosevelt National Forest of northern Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections of forest. This dataset includes information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography.
    This dataset is part of the **UCI Machine Learning Repository** ([here](https://archive.ics.uci.edu/ml/datasets/covertype)), but covertype dataset that I use comes from **Kaggle** that can be found [here](https://www.kaggle.com/uciml/forest-cover-type-dataset/). The original database owners are Jock A. Blackard, Dr. Denis J. Dean, and Dr. Charles W. Anderson of the Remote Sensing and GIS Program at Colorado State University.
* Details :

    <table style="width:100%">
    <tr>
        <td><b>Data Set Characteristics</b></td>
        <td>Multivariate</td>
    </tr>
    <tr>
        <td><b>Attribute Characteristics</b></td>
        <td>Categorical, Integer</td>
    </tr>
    <tr>
        <td><b>Associated Tasks</b></td>
        <td>Classification</td>
    </tr>
    <tr>
        <td><b>Number of Instances</b></td>
        <td>581012</td>
    </tr>
    <tr>
        <td><b>Number of Attributes</b></td>
        <td>54</td>
    </tr>
    <tr>
        <td><b>Missing Values?</b></td>
        <td>No</td>
    </tr>
    <tr>
        <td><b>Date Donated</b></td>
        <td>1998-08-01</td>
    </tr>
    </table>
* Attributes:

    | Column Name | Number of Columns |
    |---|---|
    | Elevation | 1 |
    | Aspect | 1 |
    | Slope | 1 |
    | Horizontal_Distance_To_Hydrology | 1 |
    | Vertical_Distance_To_Hydrology | 1 | 
    | Hillshade_9am | 1 | 
    | Hillshade_Noon | 1 |
    | Hillshade_3pm | 1 | 
    | Horizontal_Distance_To_Fire_Points | 1 |
    | Wilderness_Area | 4 | 
    | Soil_Type | 40 | 
    | Cover_Type | 1 |

* Data preparation :
    Weka Explorer just open data with **.arff** extension (it's like usual .csv file with header information). So, first, we have to convert **covertype.csv** into **covertype.arff** using **ArffViewer**. It can be found in the Weka's main menu > Tools > ArffViewer. 
 
## Report
### Preparation 
1. Open the dataset with **.arff** extension in the Weka
   
   ![](img/dataset-1.png)

2. 

## Classification 
### 1. Naive Bayes


### 2. K-Nearest Neighbor (KNN)
### 3. Support Vector Machine (SVM)

## Clustering 
### 4. K-Means

