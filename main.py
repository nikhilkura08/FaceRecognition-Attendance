from tkinter import *
import pandas as pd
from takeImage import take_images
from trainImage import train_images
from trackImage import track_images
import os

def save_details(id, name):
    if not os.path.exists("StudentDetails"):
        os.makedirs("StudentDetails")

    file = "StudentDetails/StudentDetails.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        df.loc[len(df)] = [id, name]
    else:
        df = pd.DataFrame([[id, name]], columns=["Id", "Name"])
    df.to_csv(file, index=False)

def start_app():
    window = Tk()
    window.title("Face Recognition Attendance System")
    window.geometry('400x250')

    lbl = Label(window, text="Enter ID")
    lbl.grid(column=0, row=0)
    txt1 = Entry(window, width=30)
    txt1.grid(column=1, row=0)

    lbl2 = Label(window, text="Enter Name")
    lbl2.grid(column=0, row=1)
    txt2 = Entry(window, width=30)
    txt2.grid(column=1, row=1)

    def take_image_cmd():
        take_images(txt1.get(), txt2.get())
        save_details(int(txt1.get()), txt2.get())

    btn1 = Button(window, text="Take Image", command=take_image_cmd)
    btn1.grid(column=0, row=3)

    btn2 = Button(window, text="Train Image", command=train_images)
    btn2.grid(column=1, row=3)

    btn3 = Button(window, text="Track Image", command=track_images)
    btn3.grid(column=1, row=4)

    window.mainloop()

if __name__ == '__main__':
    start_app()
