# -----------------------------
#   USAGE
# -----------------------------
 #python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input videos/example_activities.mp4 --gpu 1 --output output.mp4
# python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
ap.add_argument("-o", "--output", type=str, default="",	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,	help="whether or not output frame should be displayed")
ap.add_argument("-g", "--gpu", type=int, default=0,	help="whether or not it should use GPU")
args = vars(ap.parse_args())

# Load the contents of the class labels file, then define the sample (i.e., # of frames for classification) and sample
# size (i.e., the spatial dimensions of the frame)
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16    # number of frames per data point
SAMPLE_SIZE = 112       # (squared) width and height of frame

# Load the human activity recognition model
print("[INFO] Loading the human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

# check if we are going to use GPU
if args["gpu"] > 0:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Grab the pointer to the input video stream
print("[INFO] Accessing the video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = vs.get(cv2.CAP_PROP_FPS) 
print("Original FPS:", fps)

# Loop until we explicity break from it
while True:
    # Initialize the batch of frames that will be passed through the model and save original frames
    frames    = []  # frames for processing
    originals = []  # original frames
    # Loop over the number of required sample frames
    for i in range(0, SAMPLE_DURATION):
        # Read a frame from the video stream
        (grabbed, frame) = vs.read()
        # If the frame was not grabbed then we've reached the end of the video stream so exit the script
        if not grabbed:
            print("[INFO] No frame read from the stream - Exiting...")
            sys.exit(0)
        # Otherwise, the frame was read so resize it and add it to the frames list
        originals.append(frame) # save original before preprocessing
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
    # Now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                  swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    # Pass the blob through the network to obtain our human activity recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]
    # Loop over the original frames to add labels (this way we keep the original resolution)
    for frame in originals:
        # Draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
       	
        # check if the output frame should be displayed
        if args["display"] > 0:
            # show the output frame
            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')# *'MJPG' for .avi format
            writer = cv2.VideoWriter(args["output"], fourcc, fps,
                (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)
