# Tracking Cricket Ball

## Setup Steps
The following steps describe how to setup the required environement.

* Install Python.
* Create a virtual environment using:
    * python -m venv <vir_env_name>
* You need to install jupyter notebook environement in VSCode or any IDE of your choice.
* Select the python kernel from the virtual environment. 
* Install all the required packages using the following command in the terminal:
    * pip install -r requirements.txt

## Dependencies:
* ultralytics package for YOLO model.
* digitalsreeni-image-annotator package for annotating the video frames.
* supervision library for trajectory creation.

# Instructions for running the codes

* After setting up the environment , open the **data_generation.ipynb** file.
* Copy the video files into **25_nov_2025** folder (create it if it doesnt exist).
* Run each code block one by one. What this does:
    * Generates a folder named **frames** that contains multiple folders for frames of each video file in **25_nov_2025** in the format **__<video_name>__**.
    * Each **__<video_name>__** contains all the frames in that video file.
* After running the last block in the file, a window opens for the annotator software.
    * Import images from the frames folder for the video frames that needs annotations.
    * Create a class named **cricket_ball** and start annotating each frame by drawing a rectangle around the entity.
    * Save the project as **<file_name>.iap**.
    * Export the COCO JSON file for the project.
    * Combine the COCO JSONs for all the videos whose frames have been annotated.
    * Open the combined COCO JSON file and save the project as a new **.iap** file.
    * Export the project for **YOLO (v5+)**.
* You will require a powerful GPU to finetune the model (google colab is recommended). If you have a powerful GPU, ignore uploading data to google drive and mounting on google colab.
* Upload the **yolo_data** to your drive so that mounting is easy.
* Upload the **<train_model.ipynb>** file to google colab (or any GPU provider service).
* Run each code block in **<train_model.ipynb>** file.
* After running the **model.predict** block for a sample test video, you can view the model predicted video in the **runs/detect/** folder.


* To run the inference, run the following command in command line and change the arguments accordingly.

    * python modular_inference.py --input_dir <video_dir_path> --output_dir <processed_dir_path> --logs_dir <csv_dir_path> --model <model_path>.pt --imgsz <image_size> --score_th <confidence_score>

* The videos in the **<video_dir_path>** will all be processed one after the other and the processed videos will be saved in the **<processed_dir_path>** folder in .mp4 format. The required csv files will be saved inside the **<csv_dir_path>** folder.



