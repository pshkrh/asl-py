import numpy as np
import torch
import torch.nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import PIL
import cv2

#This is the Label
Labels = { 0 : 'A',
           1 : 'B',
           2 : 'C',
           3 : 'D',
           4 : 'E',
           5 : 'F',
           6 : 'G',
           7 : 'H',
           8 : 'I',
           9 : 'K',
           10: 'L',
           11: 'M',
           12: 'N',
           13: 'O',
           14: 'P',
           15: 'Q',
           16: 'R',
           17: 'S',
           18: 'T',
           19: 'U',
           20: 'V',
           21: 'W',
           22: 'X',
           23: 'Y'
        }

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0,0.225])
    ]
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation
model  = torch.load("Resnet50_Left_Pretrained_ver1.1.pth") #Load model to CPU
model  = model.to(device)   #set where to run the model and matrix calculation
model.eval()                #set the device to eval() mode for testing



#Set the Webcam
def Webcam_720p():
    cap.set(3,640)
    cap.set(4,480)

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result,score





def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    print(image)
    image = data_transforms(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor


#Let's start the real-time classification process!

cap = cv2.VideoCapture(0) #Set the webcam
Webcam_720p()

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read() #Capture each frame


    if fps == 4:
        image = frame
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
    cv2.rectangle(frame,(200,200),(0,0), (250,0,0), 2)
    cv2.imshow("ASL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL")
