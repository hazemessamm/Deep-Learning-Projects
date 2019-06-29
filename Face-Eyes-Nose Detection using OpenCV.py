import cv2 


#Importing the cascades
Face_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Eye_Cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
Nose_Cascade = cv2.CascadeClassifier('nose.xml')
def Detect(Gray, Original_Frame):
    faces = Face_Cascade.detectMultiScale(Gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(Original_Frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        
        roi_gray = Gray[y:y+h, x:x+w]
        roi_color = Original_Frame[y:y+h, x:x+w]
        eyes = Eye_Cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        
        roi_gray1 = Gray[y:y+h, x:x+w]
        roi_color1 = Original_Frame[y:y+h, x:x+w]
        nose = Nose_Cascade.detectMultiScale(roi_gray1, 1.3, 5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color1, (nx,ny), (nx+nw, ny+nh), (0, 0, 255), 2)
        
    return Original_Frame

video_capture = cv2.VideoCapture(0)
while True:
    _, Frame = video_capture.read()
    Gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    canvas = Detect(Gray, Frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
    
