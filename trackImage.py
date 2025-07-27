import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

def track_images():
    print("🟢 track_images() was called")  # Debug print

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainer.yml")
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        attendance = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

        print("▶️ Starting face tracking... Press 'q' to quit.")

        while True:
            ret, img = cam.read()
            if not ret:
                print("❌ Camera error")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)

            print("📷 Captured frame")
            print(f"🧠 Faces detected: {len(faces)}")

            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                print(f"🎯 Detected ID: {Id}, Confidence: {conf:.2f}")

                if conf < 150:
                    name = df.loc[df['Id'] == Id]['Name'].values[0]
                    now = datetime.now()
                    date = now.strftime('%Y-%m-%d')
                    time = now.strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [Id, name, date, time]
                    print(f"✅ Attendance Recorded: {Id} - {name} at {time}")
                    cv2.putText(img, f"ID:{Id} Name:{name}", (x, y-10), font, 0.8, (0, 255, 0), 2)

                else:
                    cv2.putText(img, "Unknown", (x, y-10), font, 1, (0, 0, 255), 2)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

            cv2.namedWindow("Recognizing Faces", cv2.WINDOW_NORMAL)
            cv2.imshow("Recognizing Faces", img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        if attendance.empty:
            print("⚠️ No attendance was recorded.")
            return

        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        filename = f"Attendance/Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        attendance.drop_duplicates(['ID'], keep='first', inplace=True)
        attendance.to_csv(filename, index=False)
        print(f"✅ Attendance saved to: {filename}")

    except Exception as e:
        print(f"❌ Error during tracking: {e}")
