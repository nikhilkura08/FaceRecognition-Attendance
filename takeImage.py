import cv2
import os

def take_images(Id, name):
    if not os.path.exists("TrainingImage"):
        os.makedirs("TrainingImage")

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample_num = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"TrainingImage/{name}.{Id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sample_num >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Images Saved for ID: ", Id)
