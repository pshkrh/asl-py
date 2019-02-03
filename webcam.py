import numpy as np
import cv2

#Set the WebCam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read() #Capture each frame


    if fps == 4:
        image = frame[50:300,50:300]
        cv2.imwrite('testimg.png',image)
        #image_data   = preprocess(image)
        #print(image_data)
        #prediction   = model(image_data)
        #result,score = argmax(prediction)
        fps = 0
        '''
        if result >= 0.5:
            show_res  = result
            show_score= score
        else:
            show_res   = "Nothing"
            show_score = score
        '''

    fps += 1
    #cv2.putText(frame, '%s' %(show_res),(950,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    #cv2.putText(frame, '(score = %.5f)' %(show_score), (950,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.rectangle(frame,(50,50),(300,300), (250,0,0), 2)
    cv2.imshow("ASL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL")
