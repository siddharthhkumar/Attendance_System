import cv2
import os

# Create the dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# 1. Setup Camera and Face Detector
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Input User Details
face_id = input('\n Enter Student ID (Numbers only, e.g., 101) ==>  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = 0

while(True):
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        
        # Save the captured image into the datasets folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h,x:x+w])
        
        cv2.imshow('image', img)

    # Stop after 30 photos or if 'ESC' is pressed
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30: 
         break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()