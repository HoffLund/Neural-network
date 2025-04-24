# Neural-network
Neural network to recognize digits from the MNIST dataset. It's build from absolute scratch without libraries such as tensorflow or pytorch. It was part of my SRP project in high school

# How to use and run the script(s)
These scripts are designed to be run by you! That way you can see an ai in the form of a neural network be trained in real time. 
There are two options of running the scripts: Either locally or in a Google Colab notebook (completely free).
While the second option is way easier, it's also way more limiting in terms of capabilities and gaining an understanding of the training progress.

## Option 1 - running locally:
This option requires you to download the GitHub repository as a zip-file and extract it in a folder. You need Python installed on your device and preferably and IDE too such as VS-code or PyCharm.
You can then choose if you want to run the advanced version with dynamic layers (training_5.xx_dynamic.py) or the simple version with 2 static layers (training_5.xx_(static)_simple.py). (To learn more about the difference between static and dynamic layers, read the section about it below)
While the dynamic version gives you way more options to play with, it can be a bit more difficult to understand the code, which is why i made the simple version too. This is because the data arrays and matrixes are placed in the globals so all the functions can access the data. The technical explaintion behind is that the backpropagation has to be calculated dynamicly as the amount of layers and size of the can vary. However, I am considering making a version with the same properties as the current dynamic version, but just using one big numpy array that gets bounced around between the functions. This should make the script more readable. 

## Option 2 - running in a Google Colab notebook:
This option is free, but may require you to log into your google account. However, that means you don't have to download any files to your computer. This makes it way easier to run, but at the cost of way less functions in the script as Google Colab is tricky to get continously updating output to work without it taking a 1000 years to execute.


# How is it built up?
While libraries such as Tensorflow and Pytorch basicly are used as default in machine learning, this would ruin the point of the project. I therefore built the scripts up from scratch without such libraries to help with data management, layer handling and execution of backpropagation. I did this to show that the math behind neural networks and artificial intelligence is not that complicated.


# Static and dynamic layers
What does that exactly mean? In my paper (written in danish however) I explain it in depth, but here is a brief explanation. 
When you make a neural network either on amateur scale or industrial scale you typically (very few exeptions where you do not) decide how many layers and the size of them beforehand. A generel rule of thumb is that more layers means better performance. The reason is that when you train the network and give it feedback you do backpropagation. This includes taking partial derivatives of the current layer with respect to the previous layer. Although this derivative changes depending on which layers you are taking the derivative of, what size the layers have and what functions were used in forward propagation.

However, there is a pattern in all the partial derivatives (except the last layer) that can be split into two main categories: weight-derivatives and bias-derivatives. This pattern can be seen when looking at the backpropagation function in the training_5.xx_(static)_simple.py script. This pattern can then be used to calculate all the layers dynamicly no matter how many hidden layers there are. The minimum total layers will always be two, because there needs to be and input- and output-layer. But yes! You can actually have zero hidden layers. And you might wonder how bad the network performs? Well actually not that bad. When trained with 8 epochs (time of overfitting) it had a classification percentage of a little over 92%.