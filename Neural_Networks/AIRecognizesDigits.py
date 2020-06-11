from tkinter import *
from PIL import Image, ImageGrab
from tkinter import messagebox
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Paint:
    def __init__(self):
        self.root = Tk()
        self.canvas_width = 280
        self.canvas_height = 280

        self.root.title("Neural Network Simple example")
        
        self.canvas = Canvas(self.root, width= self.canvas_width, height =self.canvas_height, bg = 'white')
        self.canvas.pack()

        self.button1 = Button(self.root, text='Submit', command= self.displayNNResult)
        self.button1.pack()

        self.button2 = Button(self.root, text='clear', command = self.clearCanvas)
        self.button2.pack()

        self.model = keras.models.load_model("mnist_model.model")

        self.canvas.bind('<B1-Motion>', self.paint)

        self.root.mainloop()

    def displayNNResult(self):
        self.save_as_jpg(self.canvas, "image1")
        messagebox.showinfo( "Hello Python", "Hello World")

    def clearCanvas(self):
        self.canvas.delete('all')

    def save_as_jpg(self, canvas, fileName):
        # save postscipt image 
        canvas.postscript(file = fileName + '.eps') 
        # use PIL to convert to PNG 
        img = Image.open(fileName + '.eps') 
        img.save(fileName + '.png') 
        
    def paint(self, event):
        circleWidth= 20
        x1, y1 = (event.x -circleWidth), (event.y -circleWidth)
        x2, y2 = (event.x + circleWidth), (event.y +circleWidth)
        self.canvas.create_oval(x1,y1,x2,y2,fill = "black", outline = "black")

    def convertImgtoBinary(self):
        print("stuff")
        #need to load image. Convert to 28 28 image. get it to binary.

if __name__ == '__main__':
    Paint()