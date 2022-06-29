import cv2


from flask import Flask, render_template
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import statistics as st

#load model
# model = model_from_json(open("fer.json", "r").read())
# #load weights
# model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# from flask import Flask, render_template
#WSGI application
app = Flask(__name__)


@app.route('/',methods = ['GET', 'POST'])
def hello():
    return render_template("index.html")

# @app.route('/camera',methods = ['GET', 'POST'])
# def camera():
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     i=0
#     output = []
#     while i<=30:
#         ret,test_img=cap.read()# captures frame and returns boolean value and captured image
#         if not ret:
#             continue
#         gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#         faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


#         for (x,y,w,h) in faces_detected:
#             cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
#             roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
#             roi_gray=cv2.resize(roi_gray,(48,48))
#             img_pixels = image.img_to_array(roi_gray)
#             img_pixels = np.expand_dims(img_pixels, axis = 0)
#             img_pixels /= 255

#             predictions = model.predict(img_pixels)

#             #find max indexed array
#             max_index = np.argmax(predictions[0])

#             emotions = ('Angry', 'Happy', 'Sad', 'Neutral')
#             predicted_emotion = emotions[max_index]
#             output.append(predicted_emotion)

#             cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#         resized_img = cv2.resize(test_img, (1000, 700))
#         cv2.imshow('Facial emotion analysis ',resized_img)



#         if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
#             break

#         i=i+1
#     cap.release()
#     cv2.destroyAllWindows()
#     final_output1 = st.mode(output)

#     cap.release()
#     cv2.destroyAllWindows
#     return render_template("index.html",final_output=final_output1,scroll = 'main')

if __name__ == '__main__':
    app.run(debug=True)
