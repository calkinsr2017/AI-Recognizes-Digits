from tkinter import *
from PIL import Image, ImageGrab
from tkinter import messagebox
import keras
import tensorflow as tf
import numpy as np

class Paint:
    def __init__(self):
       
        self.initilizeNN()

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

        self.canvas.bind('<B1-Motion>', self.paint)


        self.root.mainloop()

    def initilizeNN(self):
        data = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = data.load_data()
        self.model = tf.keras.models.load_model("mnist_model.h5")

        self.x_test = x_test

    def displayNNResult(self):
        self.save_as_jpg(self.canvas, "image1")

        self.convertImgtoBinaryAndPredict()
        #print(input)
        #result = self.makePrediction(input)
        

    def clearCanvas(self):
        self.canvas.delete('all')

    def save_as_jpg(self, canvas, fileName):
        # save postscipt image 
        canvas.postscript(file = fileName + '.eps') 
        # use PIL to convert to PNG 
        img = Image.open(fileName + '.eps') 
        img.save(fileName + '.png') 
        
    def paint(self, event):
        circleWidth= 15
        x1, y1 = (event.x -circleWidth), (event.y -circleWidth)
        x2, y2 = (event.x + circleWidth), (event.y +circleWidth)
        self.canvas.create_oval(x1,y1,x2,y2,fill = "black", outline = "black")

    def convertImgtoBinaryAndPredict(self):
        #need to load image. Convert to 28 28 image. get it to binary.
        img = tf.keras.preprocessing.image.load_img("image1.png", target_size = (28,28))
        img = np.array(img)
        input = [[] for i in range(len(img))]
        for i in range(28):
            for j in range(28):
                if (img[i][j] == [255,255,255]).all():
                    input[i].append(0)
                else:
                    input[i].append(1)
    

        for i in range(28):
            for j in range(28):
                self.x_test[0][i][j] = input[i][j]
        
        model = self.model
        predictions = model.predict(self.x_test[:1])
        t = (np.argmax(predictions[0]))
        messagebox.showinfo("Response", "I do believe thats a " + str(t))
        
    


if __name__ == '__main__':
    Paint()
    

    