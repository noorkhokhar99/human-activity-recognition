# Human Activity Recognition with OpenCV and Deep Learning
- video for [Human Activity Recognition with OpenCV and Deep Learning](https://www.youtube.com/c/Pyresearch/videos) tutorial.

## Additional Notes
- Downloaded the ONNX model as per download_models.py from [HERE resnet-34_kinetics.onnx](https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1')
- Human Activity Recognition model requires at least OpenCV 4.1.2.
# human-activity-recognition




### Steps to run Code
- Clone the repository.
```
git clone https://github.com/noorkhokhar99/human-activity-recognition.git
```
- Goto the cloned folder.
```
cd human-activity-recognition

```
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Run the code with mentioned command below.
```
# if you want to run source file
 python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input videos/example_activities.mp4 --gpu 1 --output output.mp4
 
 python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

```

### Results


<img src="https://github.com/noorkhokhar99/human-activity-recognition/blob/main/Screen%20Shot%201444-04-07%20at%202.03.38%20AM.png">

<img src="https://github.com/noorkhokhar99/human-activity-recognition/blob/main/Screen%20Shot%201444-04-07%20at%202.03.33%20AM.png">

<img src="https://github.com/noorkhokhar99/human-activity-recognition/blob/main/Screen%20Shot%201444-04-07%20at%202.03.15%20AM.png">
<img src="https://github.com/noorkhokhar99/human-activity-recognition/blob/main/Screen%20Shot%201444-04-07%20at%202.02.48%20AM.png">


