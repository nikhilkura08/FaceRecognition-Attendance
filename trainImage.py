import cv2
import numpy as np
from PIL import Image
import os

def train_images():
    try:
        print("📢 train_images() was called")  # ✅ Debug line added
        print("🔁 Starting training...")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'TrainingImage'
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            ids = []

            for image_path in image_paths:
                try:
                    pil_img = Image.open(image_path).convert('L')  # Convert to grayscale
                    img_np = np.array(pil_img, 'uint8')
                    id = int(os.path.split(image_path)[-1].split(".")[1])
                    faces.append(img_np)
                    ids.append(id)
                except Exception as e:
                    print(f"⚠️ Skipping {image_path}: {e}")

            return faces, ids

        faces, ids = get_images_and_labels(path)

        if len(faces) == 0:
            print("❌ No training images found in 'TrainingImage/' folder.")
            return

        recognizer.train(faces, np.array(ids))

        if not os.path.exists("TrainingImageLabel"):
            os.makedirs("TrainingImageLabel")

        recognizer.save("TrainingImageLabel/Trainer.yml")
        print("✅ Training Completed Successfully")

    except Exception as e:
        print(f"❌ Error during training: {e}")
