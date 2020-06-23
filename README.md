# AI Recognizes Digits 

Using a simple Tkinter GUI and tensorflow I was able to build an AI that recognizes user hand written digits. The simple gui allows the user to draw a digit with a mouse or touch screen. The neural network model is trained using the MNIST dataset provided by keras. 

## Getting Start

Clone the reposiitory and pip install the requirements, not much more to it! If you want to train your own model, run NN_MNIST.py making sure the retrain function is set to true. This takes a while as I am preprocessing the training data set to look like the user input. 

To run the GUI, simply run AIRecognizesDigits.py. I use xming as my local host from ubuntu to display the GUI. After you draw your single digit, click "submit" for the model to  guess the digit. To restart, click "clear". It should be noted that, it is a subpar trained model and has trouble guessing less then perfect digits. See example results below:
<p align="center">
<img src="https://github.com/calkinsr2017/AI-Recognizes-Digits/blob/master/Images/six.JPG" width="300" height="200">                    <img src="https://github.com/calkinsr2017/AI-Recognizes-Digits/blob/master/Images/four.JPG" width="300" height="200">
</p>

## Built with

TKinter, TersorFlow and numpy are the main modules used. I took inspiration from <a href="https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg">TechWithTim's</a> youtube series on ML. The dataset was loaded and processed by keras.

## Moving forward

This was my first ML project, while I am proud of the final result there are still many improvements that can be made as I learn more about ML. The model has low accuracy when the input digits are less then perfect, and still has trouble recognizing seven's and nine's. I did a small test to see how accurate the model was with each digit. While still attempting to draw decent digits with small variations, the model had the following accuracy:

<p align="center">
<img src="https://github.com/calkinsr2017/AI-Recognizes-Digits/blob/master/Images/meta-chart.png" width="500" height="300">
</p>
  
Since I trained the model using the MNIST dataset and not my own, the model is not used to the format I have given it. As I continue to learn about ML, I will come back to this project in order to gradually retrain the model with my own generated data.
