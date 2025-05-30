# Traffic Stats

This repository contains all the files used during the experimentation and development for the project named Traffic Stats.

## Files and directories
- training_for_cars/ - code used to fine tune YOLO with custom dataset.
- traffic_stats.py, state_init.py, NoBufferVC.py, Model.py, get_stream.py - code used for the monitoring and collection of data
- data_collected/ - collected JSON data
- data_fixing/ - code used to fix some JSON files where detections would disappear for one frame
- data_analysis/ - code used to visualize the distribution of the collected data
- generator_model/ - code used to train and test the simple and FFNN models with the collected data
- sumo/ - unused visualization tests using SUMO
- dtprototype.py - Digital Twin Prototype Application. Reuses some of the assets used by the monitoring and collection app
- gradio_test/ - gradio UI before switching to Streamlit