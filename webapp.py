#!/usr/bin/env python
from pprint import pprint as pp
from flask import Flask, flash, redirect, render_template, request, url_for, Response
from googletrans import Translator
import cv2

app = Flask(__name__)
classNames = []
classFile = 'coco.names'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
cap = cv2.VideoCapture(0)
thres = 0

@app.route('/')
def index():
    return render_template(
        'weather.html',
        data=[{'name':'German'}, {'name':'French'}, {'name':'Spanish'}])

def gen(cap):
    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):

                confidenceValue_str = str(round(confidence * 100))
                confidenceValue = int(confidenceValue_str)

                if (confidenceValue >= 60):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId - 1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

        
        ret, jpeg = cv2.imencode('.jpg', img)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed', methods=["GET", "POST"])
def video_feed():
    selection = request.form.get('comp_select')
    selection_String = str(selection)

    if selection_String == "German":
        langSelect("de")
    elif selection_String == "French":
        langSelect("fr")
    else:
        langSelect("es")
        
    videoSetup()
    global cap
    return Response(gen(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def langSelect(lang) -> None:
    googTransl = Translator()
    with open(classFile,'rt') as f:
        for line in f:
            string = str(line.strip())
            googTranslResult = googTransl.translate(string, dest=lang)
            new_lang = googTranslResult.text
            classNames.append(new_lang)

def videoSetup() -> None:
    thres = 0.45
    cap.set(3,1280)
    cap.set(4,720)
    cap.set(10,70)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)