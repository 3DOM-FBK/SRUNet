This folder contains files related to the SRUNet model
First, need to install the required libraries which are mentioned in the file “requirements.txt”
This model was trained using tensorflow/keras

For training, user can compile the file “SRUNet_100_50.py” for the Super Resolution from 1m to 50cm, and “SRUNet_50_25.py” for the Super Resolution from 50cm to 25cm

**For prediction, user can use “predict_100_50_25.py” to predict the result of SR from 1m to 25cm using the 2 models (1m->50cm then 50cm->25cm). 

the file "convert_npy_tif.py" is to convert the result of prediction from python numpy format (npy) to "geotif" format
the file 'convert_tif_npy.py' is to convert the origin tiff file from tif to python numpy data (npy)

Because of issue of disk-space, the training data and the trained models are uploaded separately
------The links to training data--------
# for data of SR from 1m to 50cm
https://drive.google.com/file/d/1Ck9ZLwTSudAoHbtlBy8IUuvPBrrfEvFH/view?usp=sharing
# for data of SR from 50cm to 25cm
https://drive.google.com/file/d/1AskFcMv96OapsFVylBq6dOH4qHionesq/view?usp=sharing

-----link to the SRUNet models------
# SRUNet_100_50 model
https://drive.google.com/file/d/1u4Hs23VLRvTJPrWNY16sdu4sJJgv4arM/view?usp=sharing
# SRUNet_50_25 model
https://drive.google.com/file/d/1dJzfz4G7bAAhwo5DuxThQpIKXwG30wvM/view?usp=sharing

-----link to the testing data------
https://drive.google.com/drive/folders/1oxZgobCVM-MmQtnqmk2Ju9YxA4ld5r2y?usp=sharing