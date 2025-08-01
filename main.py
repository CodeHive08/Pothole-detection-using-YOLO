from ultralytics import YOLO
import cv2
import numpy as np  
import cvzone

model=YOLO("best.pt")
class_names=model.names
cap=cv2.VideoCapture("video.mp4")
count=0
while True:
    ret,img=cap.read()
    if not ret:
        break
    count+=1
    if count%3 != 0:
        continue
    img=cv2.resize(img,(1020,500))
    h,w,_=img.shape
    results=model.predict(img)
    for r in results:
        boxes=r.boxes
        masks=r.masks
    if masks is not None:
        masks=masks.data.cpu()
        detected_boxes=[]
        for seg,box in zip(masks.data.cpu().numpy(),boxes):
            seg=cv2.resize(seg,(w,h))
            contours,_=cv2.findContours(seg.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
               x, y, bw, bh = cv2.boundingRect(contour)
               overlap=False
               for bx,by,bbw,bbh in detected_boxes:
                    if (x < bx + bbw and x + bw > bx and
                        y < by + bbh and y + bh > by):
                        overlap = True
                        break
               if not overlap:
                    detected_boxes.append((x, y, bw, bh))
                    d = int(box.cls)
                    c = class_names[d]
                    points = contour.reshape(-1, 2).tolist()
                    cv2.polylines(img, [np.array(points, np.int32)], True, (255,0,0), 2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0,0,255), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows() 
# Save the processed video if needed
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1020, 500))
# out.write(img)
# out.release()
