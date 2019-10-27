### Assignment 3
# Predicting Residential Energy Consumption Using CNN-LSTM Neural Networks

> Hafara Firdausi
> 05111950010040

## Implementation Environment
This program implemented in the system described below:
* Operating System : MacOS Mojave Version 10.14.5
* Memory : 8 GB 1600 MHz DDR3
* Processor : 2.6 GHz Intel Core i5 (I5-4278U)
* Python Environment :
  * Python 3.7.3
  * Keras 2.2
  * TensorFlow/Theano
  * Scikit-learn
  * Pandas
  * Numpy
  * Matplotlib
  * Jupyter Notebook

## Problem Description 
* This problem using a dataset from UCI Machine Learning named **"Individual Household Electric Power Consumption Dataset"**, a multivariate time series dataset contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (3 years 11 months). 
* It has 9 attributes described below:

    | Attributes | Description |
    |---|---|
    | **date** | Date (dd/mm/yyyy) | 
    | **time** | Time (hh:mm:ss)| 
    | **global_active_power** | The total active power consumed by the household (Kilowatt) |
    | **global_reactive_power** | The total reactive power consumed by the household (Kilowatt) |
    | **voltage** | Average voltage (Volt) |
    | **global_intensity** | Average current intensity (Ampere) |
    | **sub_metering_1** | Active energy for kitchen, containing mainly a dishwasher, an oven and a microwave (Watt-hour of active energy) |
    | **sub_metering_2** | Active energy for laundry rooms, containing a washing-machine, a tumble-drier, a refrigerator and a light (Watt-hour of active energy) |
    | **sub_metering_3** | Active energy for climate control, containing an electric water-heater and an air-conditioner systems (Watt-hour of active energy)|

## Steps

* [Load and Prepare Dataset](load-and-prepare-dataset.ipynb)


## References
* [T.-Y. Kim and S.-B. Cho, "Predicting Residential Energy Consumption Using CNN-LSTM Neural Networks," Energy, vol. 182, pp. 72-81, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S0360544219311223)
* [“How to Develop Multi-Step LSTM Time Series Forecasting Models for Power Usage,” *Machine Learning Mastery*, 10-Oct-2018. [Online]. Available: https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/. [Accessed: 12-Oct-2019].](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/)