# RockPaperScissorsML
Using a NN to play Rock Paper Scissors

Im currently going through a bunch of youtube tutorials. Using Sklearn and tensorflow to create a bunch of different ML models. 
We can see if we wanna do any of them from scratch. Just dicking around and learning about the different model types. 

The tutorial I am following
https://www.youtube.com/watch?v=ujTCoH21GlA&t=19s 

### Linear Regression
Make sure to set RETRAIN to true before running LinearRegression.py to generate your best fit model.
Set it to false afterwards

Use linear regression when data has a noticable correlation. AKA MX+B best fit line situation. 

### k Nearest neighbor
No idea what the hell this is....

### Neural Networks!!!!
Recognizing digits or clothing or whatever
our data can be images that are 28X28 pixels. Each pixel will have 3 values for RGB.
ex. [[10, 32, 131, 12...][..][..]] 28x28 2d array. 3 of them.
In order to use a NN we will need to flatten the data. get 784x1 list. 
input layer will have 784 bubbles.Output will will have 0-9.
Hidden layer can be anything. Good to go off percentage of previous layer. For this 128.

### Brainstorming section
1. Handwritten digits. Have a pop-up box where the user can draw a digit and the NN will guess it
2. Clone a game (Snake) and have the NN automate it
3. Live Rock paper scissors (Make the computer always win be recognizing hands super fast)
4. Automate tesla tickle (https://www.urbandictionary.com/define.php?term=Tesla%20Tickle)


Plan of action! I, Robert, am going to make a handwritten digit recognizer. I will use either, pygame, TKinter or some other method
to draw on a canvas. Then use ImageGrab to get a 28x28 pic to get into the model. 
