from core import *
from Tkinter import *


class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.button = Button(frame,
                             text="Process from Webcam", fg="red",
                             command=self.webcam_click, width=50)
        self.button.pack(side=LEFT)
        self.slogan = Button(frame,
                             text="Process from File",
                             command=self.file_click, width=50)
        self.slogan.pack(side=LEFT)

    def webcam_click(self):
        process_from_cam()

    def file_click(self):
        process_from_file()






root = Tk()
root.title("Virtual Reality FEUP")
app = App(root)
root.mainloop()