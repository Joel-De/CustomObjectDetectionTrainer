# CustomObjectDetectionTrainer
 Complete end to end pipeline designed for users to train their own object detection machine version model.


### Data Capture:
1. Within the DataLabeler Folder run ImageRecorder, this will take pictures with your webcam and save them to a folder, if you have your own pictures copy and paste them into the Images folder and skip this step
2. Download the Labeling Tool here https://tzutalin.github.io/labelImg/ Label the images saving the XML files in a folder called Labels
3. Run the Convert Script, this will convert the XML files into jsons parseable by the train script


### Train:
1. Configure the ObjectDetectionConfig.json file with the appropriate settings i.e. entities directory names, the Dataset directory is simply a directory containing both the Images and Json folder.
2. Run the train script, depending on your hardware this may take some time. If a GPU is available on your system it will be used. As of now Mult-GPU is not supported.

### Inference:
1. Create an instance of the Predictor class, examples of how to use it are in the main function. The Predict function takes an image and returns a dictionary corresponding with class names and bounding box location

