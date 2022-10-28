from flask import Flask, Response
import cv2

app = Flask(__name__)
video = cv2.VideoCapture(1)
#face_cascade = cv2.CascadeClassifier()
#face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))

@app.route('/')
def index():
    return "Default Message"

def gen(video):
    while True:
        success, image = video.read(1)
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        #faces = face_cascade.detectMultiScale(frame_gray)

        
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)