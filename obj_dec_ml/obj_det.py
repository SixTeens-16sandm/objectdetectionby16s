import cv2
import matplotlib.pyplot as plt
import pyttsx3

def convert_output_to_sentence(output):
    counts = {}
    positions = {}
    sentence_list = output.split('\n')
    for i in range(len(sentence_list)-1):
        obj_pos_list = sentence_list[i].split(' - ')
        obj = obj_pos_list[0]
        pos = obj_pos_list[1]
        if obj in counts:
            counts[obj] += 1
        else:
            counts[obj] = 1
        if obj in positions:
            if pos in positions[obj]:
                positions[obj][pos] += 1
            else:
                positions[obj][pos] = 1
        else:
            positions[obj] = {pos: 1}
    
    sentences = []
    for obj, count in counts.items():
        sentence = f"There {'are' if count > 1 else 'is'} {count} {obj}"
        if count > 1:
            sentence += "s"
        sentence += ","
        pos_count = positions[obj]
        if 'center' in pos_count:
            sentence += f" {pos_count['center']} at center"
            if 'right' in pos_count or 'left' in pos_count:
                sentence += " ,"
            else:
                sentence += "."
        if 'right' in pos_count:
            sentence += f" {pos_count['right']} at right"
            if 'left' in pos_count:
                sentence += " ,"
            else:
                sentence += "."
        if 'left' in pos_count:
            sentence += f" {pos_count['left']} at left"
        sentences.append(sentence)
    sentence += "."
    
    return ", ".join(sentences)


text_speech = pyttsx3.init()

config_file = "obj_dec/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "obj_dec/frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = "obj_dec/labels.txt"
with open(file_name) as file:
    classLabels = file.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)

# TODO: You have to change the below image path
img = cv2.imread('obj_dec/boy.jpeg') 

# Find image width
image_width = img.shape[1]
print("width", image_width)

left_threshold = image_width/3
center_threshold = left_threshold*2
right_threshold = image_width

ClassIndex, confidece, bbox = model.detect(img, confThreshold=0.5)
print(ClassIndex)

count = 1
sentence = ""

# Annotated image with location information
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    
    x_min = boxes[0]
    if(x_min<left_threshold):
        location = "left"
    elif (x_min<center_threshold):
        location = "center"
    else:
        location = "right"
        
    sentence += f"{classLabels[ClassInd-1]} - {location}\n"
    count+=1

final_sentence = convert_output_to_sentence(sentence)
print(final_sentence)
text_speech.say(final_sentence)
text_speech.runAndWait()

