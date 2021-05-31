# CustomObjectDetectionTrainer
 Complete end to end pipeline designed for users to train their own object detection machine version model, from data collection to inference.


Data Capture:
1. Within the DataLabeler Folder run ImageRecorder, this will take pictures with your webcam and save them to a folder, if you have your own pictures copy and paste them into the Images folder
2. Download the Labeling Tool here https://tzutalin.github.io/labelImg/ Label the images saving the XML files in a folder called Labels
3. Run the Convert Script


Train:
1. Configure the ObjectDetectionConfig.json file with the appropriate settings ie. entities directory names, the Dataset directory is simply a directory containing both the Images and Json folder.
2. Run the train script

Inference:
1. Create and instance of the Predictor class, examples of how to use it are in the main function. The Predict function takes an image and returns a dictionary corrosponding with class names and bounding box location

