# PatchTSMixer

PatchTSMixer ([link](https://arxiv.org/abs/2306.09364)) is a state-of-the-art time-series model designed for multi-variate time-series forecasting and classification. With this model, the classification task is thus to prediction one binary label out of a 420-step sequence. 


## Usage
First preprocess the extracted gaze event data at [my_data/processed_gaze_21s](../my_data/processed_gaze_21s/) using [data/CPR_simple_preprocessing.py](data/CPR_simple_preprocessing.py). As mentioned in the paper, one can set `use_gaze_behavior=False` to not use gaze event at each timestamp as a input feature. This should generate respective time-series data in [my_data/processed_ts_21s/{features}](../my_data/processed_ts_21s).

Once the data is prepared, use the following command to train and evaluate PatchTSMixer with the default configuration (same as what we reported in the paper):
```
python patchtsmixer.py --data_dir ../my_data/processed_ts_21s/center_event 
```
Alternatively, you may play with configurations such as `d_model`, `patch_length`, etc. to train your own model. We recommand using some sort of normalization but it also depends on your specific data.

The results will be written to tensorboard for your review.