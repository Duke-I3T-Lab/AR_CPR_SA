# AR_CPR_SA
This is the official code repository for the ISMAR 2025 paper "Will You Be Aware? Eye Tracking-Based Modeling of Situational Awareness in Augmented Reality", authored by Zhehan Qu, Tianyi Hu, Christian Fronk and Maria Gorlatova.

## Overview
![Overview](assets/teaser.png)
This work seeks to model situational awareness though eye tracking data captured on a Magic Leap 2 device, in an AR app designed for cardiopulmonary resuscitation (CPR) guidance. To evaluate situational awareness, we designed two realistic, unexpected incidents, i.e. the patient bleeding or vomiting during the CPR procedure, to observe the participants' response. Based on their responses we label them with good or poor SA labels and trained a graph neural network that predicts such label at 83% accuracy. Below 

## Video Demostration
Check the video below for how we setup our experiment and how the incidents look in the AR view. (Video link temporarily left blank before final decision)
<p align="center">
    <a href="">
        <img src="assets/example.png" alt="Watch the video" style="width:50%;">
    </a>
</p>

## Code Base Introduction
The code base consists of three main modules: the [gaze extraction and classic ML module](gaze_extraction), the (baseline) [PatchTSMixer module](time_series_modeling) (thanks to George Zerveas et al. for open sourcing their [code](https://github.com/gzerveas/mvts_transformer), from which we built a part of the model training/testing workflow), and the [FixGraphPool module](fix_graph_pool_modeling). The recommanded workflow (and so was our workflow) is to first compile raw collected gaze data through the gaze extraction module, and then run preprocessing code on gaze extraction output to run the different models. 

**Please note that due to IRB requirements we are not able to release the data.** Instead, we provide *sample data files* in [my_data](my_data) to showcase how the raw data and preprocessed data should look like if the code is used properly. Please be aware that all these files are randomly generated and nothing should be implied from the values provided. 

For more details of code usage, please refer to seperate README files in each module. 
- [gaze extraction and classic ML module](gaze_extraction/README.md)
- [PatchTSMixer module](time_series_modeling/README.md)
- [FixGraphPool module](fix_graph_pool_modeling/README.md)


## Citation
Citation temporarily left blank before final decision.