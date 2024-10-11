import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import torch

# Kiểm tra xem GPU có khả dụng không
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU configuration.")
else:
    print("CUDA is available.")
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(0))

# Khởi tạo mô hình YOLO và chuyển sang GPU
model = YOLO('yolov8m.pt').to('cuda')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('Parking Car')
cv2.setMouseCallback('Parking Car', RGB)

cap=cv2.VideoCapture('parking1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

area1=[(298,40),(350,38),(348,92),(283,90)]   

area2=[(353,38),(415,40),(420,93),(349,92)]

area3=[(415,39),(479,41),(498,96),(421,92)]

area4=[(480,41),(550,46),(574,102),(499,96)]

area5=[(550,45),(617,47),(660,110),(582,101)]

area6=[(618,50),(690,55),(738,118),(660,110)]

area7=[(697,56),(758,63),(815,128),(742,118)]

area8=[(283,94),(344,95),(340,169),(265,165)]

area9=[(348,94),(420,96),(430,176),(344,167)]

area10=[(422,97),(501,99),(522,180),(432,174)]

area11=[(500,99),(578,105),(619,191),(525,182)]

area12=[(578,107),(659,111),(714,199),(621,192)]

area13=[(665,113),(741,120),(806,210),(720,199)]

area14=[(745,123),(820,131),(890,222),(806,208)]

area15=[(246,280),(337,285),(366,437),(247,433)]

area16=[(337,285),(435,286),(500,438),(366,436)]

area17=[(500,290),(600,293),(711,423),(579,432)]


while True:    
    ret,frame = cap.read()
    if not ret:
        break
    time.sleep(0.01)
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    a=results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")
    
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    list7=[]
    list8=[]
    list9=[]
    list10=[]
    list11=[]
    list12=[]
    list13=[]
    list14=[]
    list15=[]
    list16=[]
    list17=[]
    
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2

            results1=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if results1>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list1.append(c)
               cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            
            results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if results2>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list2.append(c)
            
            results3=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            if results3>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list3.append(c)   
            results4=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
            if results4>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list4.append(c)  
            results5=cv2.pointPolygonTest(np.array(area5,np.int32),((cx,cy)),False)
            if results5>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list5.append(c)  
            results6=cv2.pointPolygonTest(np.array(area6,np.int32),((cx,cy)),False)
            if results6>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list6.append(c)  
            results7=cv2.pointPolygonTest(np.array(area7,np.int32),((cx,cy)),False)
            if results7>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list7.append(c)   
            results8=cv2.pointPolygonTest(np.array(area8,np.int32),((cx,cy)),False)
            if results8>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list8.append(c)  
            results9=cv2.pointPolygonTest(np.array(area9,np.int32),((cx,cy)),False)
            if results9>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list9.append(c)  
            results10=cv2.pointPolygonTest(np.array(area10,np.int32),((cx,cy)),False)
            if results10>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list10.append(c)     
            results11=cv2.pointPolygonTest(np.array(area11,np.int32),((cx,cy)),False)
            if results11>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list11.append(c)    
            results12=cv2.pointPolygonTest(np.array(area12,np.int32),((cx,cy)),False)
            if results12>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list12.append(c)
            results13=cv2.pointPolygonTest(np.array(area13,np.int32),((cx,cy)),False)
            if results13>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list13.append(c)  
            results14=cv2.pointPolygonTest(np.array(area14,np.int32),((cx,cy)),False)
            if results14>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list14.append(c)  

            results15=cv2.pointPolygonTest(np.array(area15,np.int32),((cx,cy)),False)
            if results15>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list15.append(c)  

            results16=cv2.pointPolygonTest(np.array(area16,np.int32),((cx,cy)),False)
            if results16>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list16.append(c)  

            results17=cv2.pointPolygonTest(np.array(area17,np.int32),((cx,cy)),False)
            if results17>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list17.append(c) 

              
    a1=(len(list1))
    a2=(len(list2))       
    a3=(len(list3))    
    a4=(len(list4))
    a5=(len(list5))
    a6=(len(list6)) 
    a7=(len(list7))
    a8=(len(list8)) 
    a9=(len(list9))
    a10=(len(list10))
    a11=(len(list11))
    a12=(len(list12))
    a13=(len(list13))
    a14=(len(list14))
    a15=(len(list15))
    a16=(len(list16))
    a17=(len(list17))
    o=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15+a16+a17)
    space=(17-o)
    print(space)

    if a1==1:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('1'),(323,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('1'),(323,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a2==1:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('2'),(380,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('2'),(380,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a3==1:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('3'),(446,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('3'),(446,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a4==1:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('4'),(510,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('4'),(510,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a5==1:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('5'),(580,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('5'),(580,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a6==1:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('6'),(644,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('6'),(644,29),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
    if a7==1:
        cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('7'),(720,39),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('7'),(720,39),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a8==1:
        cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('8'),(300,191),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('8'),(300,191),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)  
    if a9==1:
        cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('9'),(398,198),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('9'),(398,198),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a10==1:
        cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('10'),(478,200),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('10'),(478,200),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a11==1:
        cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('11'),(572,206),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('11'),(572,206),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a12==1:
        cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('12'),(666,210),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('12'),(666,210),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a13==1:
        cv2.polylines(frame,[np.array(area13,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('13'),(769,228),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area13,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('13'),(769,228),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a14==1:
        cv2.polylines(frame,[np.array(area14,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('14'),(854,237),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area14,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('14'),(854,237),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    if a15==1:
        cv2.polylines(frame,[np.array(area15,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('15'),(290,258),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area15,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('15'),(290,258),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    if a16==1:
        cv2.polylines(frame,[np.array(area16,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('16'),(382,259),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area16,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('16'),(382,259),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    if a17==1:
        cv2.polylines(frame,[np.array(area17,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('17'),(544,267),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area17,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('17'),(544,267),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    cv2.putText(frame,str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("Parking Car", frame)

    key = cv2.waitKey(1)
    
    
    if key == 32:  # Nếu nhấn phím Space tua video qua 10 phút
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        new_time = current_time + 600000 
        cap.set(cv2.CAP_PROP_POS_MSEC, new_time)

    if key & 0xFF == 27:  # Nhấn phím Esc để thoát
        break

cap.release()
cv2.destroyAllWindows()
