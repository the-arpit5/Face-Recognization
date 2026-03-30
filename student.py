import cv2
import sqlite3

def capture_data():
    student_id = input("Enter Student ID: ")
    name = input("Enter Student Name: ")
    
    # Camera open karna
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    count = 0
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            # Data folder mein photos save karna
            cv2.imwrite(f"data/user.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        cv2.imshow("Capturing Photos...", frame)
        if cv2.waitKey(1) == 27 or count == 100: 
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Photos Saved in 'data' folder!")

if __name__ == "__main__":
    capture_data()