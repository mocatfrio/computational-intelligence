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
  
   1. Weka Explorer just open data with **.arff** extension (it's like usual .csv file with header information). So, first, we have to convert **covertype.csv** into **covertype.arff** using **ArffViewer**. It can be found in the Weka's main menu > Tools > ArffViewer. Open the .csv file and save as .arff file.

    ![](img/arffviewer.png)

   2. There are **too much data** that cause some processes in Weka to get stuck. So, I reduce the number of data become **100000 rows**. (source: [here](https://stackoverflow.com/questions/50820926/weka-j48-gets-stuck-on-building-model-on-training-data) | data reduction process: [here](reduce-covertype.ipynb))
 
## Report
### Preprocess 
1. Open the dataset with **.arff** extension in the Weka
   
   ![](img/dataset.png)

2. Some algorithms must use data with **Nominal** type. So, we must preprocess the data first by using filter **NumericToNominal**.

    ![](img/preprocess-numeric-to-nominal.png)

3. Click **Visualize all** to visualize all attributes.
   
   ![](img/visualize-1.png)
   
   ![](img/visualize-2.png)
   
### Classification - Naive Bayes
1. Click **Classify** tab.
2. Click **Choose** button and select **NaiveBayes** under the **bayes** group.
   
   ![](img/naive-bayes.png)

3. Click the **Start** button to run the algorithm on the Covertype dataset.
4. Here's the result
   
   ![](img/naive-bayes-result-1.png)

   ![](img/naive-bayes-result-2.png)

    You can see that with the default configuration that Naive Bayes achieves an accuracy of **77%**.

### Classification - K-Nearest Neighbor (KNN)

1. Still in the **Classify** tab.
2. Click the **Choose** button and select **IBk** under the **lazy** group.

   ![](img/knn.png)

   ![](img/nn.png)

3. Click the **Start** button to run the algorithm on the Covertype dataset.
4. Here's the result

   ![](img/knn-result-1.png)

   ![](img/knn-result-2.png)

    You can see that with the default configuration that KNN achieves an accuracy of **69%**.

### Classification - Support Vector Machine (SVM)

1. Still in the **Classify** tab.
2. Click the **Choose** button and select **SMO** under the **function** group.

   ![](img/svm.png)

3. Click the **Start** button to run the algorithm on the Covertype dataset.
4. Here's the result

   ![](img/svm-error.png)

   It still error because out of memory.

### Clustering - K-Means

1. Click **Cluster** tab.
2. Click the Clusterer “Choose” button and select .
3. Click the **Choose** button and select **SimpleKMeans** under the **clusterers** group.

   ![](img/kmeans.png)

4. Click the **Start** button to run the algorithm on the Covertype dataset.
5. Here's the result

   ![](img/kmeans-result.png)

