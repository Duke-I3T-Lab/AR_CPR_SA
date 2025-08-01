# Gaze Extraction and Classic Machine Learning

This module consists of code that (1) extractis gaze events from raw gaze data and (2) processes data to windows of features and conducts classic machine learning on it. 

## Gaze Event Extraction
Run `python cpr_main_extraction.py` to extract gaze events and store them in preprocessable format. There are several configurable parameters, including:
- `window size` for velocity computation: for 60Hz data we recommend set it to 3
- `window size` for mobility detection: this is for detecting head movement as we described in the paper, window size 5 suits better as the scale of movement is larger than that of gaze movements
- other hyperparameters were selected based on the [Tobii recommendations](https://connect.tobii.com/s/article/Gaze-Filter-functions-and-effects?language=en_US) on hyperparameter selection for I-VT algorithm

This file should output processed gaze data into [my_data/processed_gaze_21s](../my_data/processed_gaze_21s/). It should be in the format of file pairs of `.csv` and `.json` files: the `.csv` file should contain raw gaze data with gaze event annotations, while the `.json` file should contain a dictionary of fixations, saccades and blinks, in which each event should have a number of attributes as defined in [GazeDataExporter](offline/modules/computation.py#L580).

## Classic Machine Learning
[data_prepare_ml_cpr.py](data_prepare_ml_cpr.py) contains the code that generates features of fixations, saccades, blinks (and pupil dilation, although we did not end up using it) and ROI-related stats of windows of 7s of data. These data will be saved to [my_data/processed_ml_21s](../my_data/processed_ml_21s/) for further training and testing. 

[prediction/machine_learning_cpr_all.py](prediction/machine_learning_cpr_all.py) contains the code that trains different ML models for all 5 folds and summarizes the result in a printed table. One can also customize their preferred feature set [here](prediction/machine_learning_cpr_all.py#L38) and configure grid search parameters [here](prediction/machine_learning_cpr_all.py#L49)