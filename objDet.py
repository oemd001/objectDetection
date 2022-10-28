import cv2
from googletrans import Translator

#global vars
classNames= []
classFile = 'coco.names'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
cap = cv2.VideoCapture(1)
thres = 0

class objDet():
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

    def run() -> None:
        for i in classNames:
            print(i)

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
        
            cv2.imshow("Output",img)
            cv2.waitKey(1)

    def runEverything(lang) -> None:
        objDet.langSelect(lang)
        objDet.videoSetup()
        objDet.run()