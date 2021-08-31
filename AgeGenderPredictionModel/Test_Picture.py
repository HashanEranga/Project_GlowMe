import os
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import FaceDetectionModule
#%%%%

test_path='testing/Team'   #Images Path
#test_path='testing/Test'

img_names=os.listdir(test_path)
category_dict={0:'Female',1:'Male'}

import pickle
dict_a=pickle.load(open("age_labels.data","rb"))
#print(dict_a)

#%%
from keras.models import load_model
# model = load_model('models')
# model_age=load_model('mod_age')
model=load_model('model_age_gender.hdf5')
detector = FaceDetectionModule.FaceDetector(minDetectionCon = 0.2)
#model.summary()
#%%
for img_name in img_names:
    img=cv2.imread(os.path.join(test_path,img_name))
    #print(img.shape[0])
    if img.shape[0]> 800:
        img = imutils.resize(img, width=600)
    #img=cv2.resize(img,(800,800))
    h1,w1 = img.shape[0:2]
    img ,bboxs = detector.findFaces(img,False)
    print(bboxs)
    for box in bboxs:
        print(box[1])
        x,y,w,h = box[1]
        test_img = img[x:x+w,y:y+h]
        print(test_img.size)
        if test_img.size<=0:
            continue
        test_img = cv2.resize(test_img,(224,224))
        test_img = test_img.reshape(1,224,224,3)
        results=model.predict(test_img)
    
        predicted_genders = results[0][0]
        gen=np.argmax(predicted_genders)
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        #img[y+h:y+h+min(40,(h//3)),x:x+w+min(30,(w//5))]=[162,42,42]
        #img[y-min(40,(h//3)):y,x:x+w+min(30,(w//5))]=[0,255,0]
        img[y+h:y+h+min(40,(h//3)) ,x:x+w+2]=[162,42,42]
        img[y-min(40,(h//3)):y,x:x+w+2]=[0,255,0]
        gender_predict=category_dict[gen]
        #gender_predict="Female" #Maneul
        cv2.putText(img,gender_predict,(x+(w//25),y-min(10,(h//10))),cv2.FONT_HERSHEY_SIMPLEX,min(1.2,(h/100)),(255,255,255),2)
        #cv2.putText(img,str(acc)+'%',(x+w-70,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)   
        cv2.putText(img,str(int(predicted_ages)),(x+(w//25),y+h+min(35,(h//4))),cv2.FONT_HERSHEY_SIMPLEX,min(1.2,(h/100)),(255,255,255),2)
        #cv2.putText(img,str(acc_age)+'%',(x+w-70,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2)
         
        #img=cv2.resize(img,(0,0),fx=0.6,fy=0.6)
    cv2.imshow('LIVE',img)
    #cv2.imwrite("CC14.jpg", img)
    k=cv2.waitKey()
    if(k==27):
        break
    #img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #plt.imshow(img1)
    #plt.savefig("CC.png", dpi=500)
    
cv2.destroyAllWindows()
    

