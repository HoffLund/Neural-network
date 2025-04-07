import os                           # used for clearing the save folder when a new save is created
import numpy as np                  # used for the matrix operations and matrix structure
import matplotlib as mpl            # used for coloring the images
import matplotlib.pyplot as plt     # used for plotting the loss-graph and showing the wrong images
import time                         # used to track the training time

### Fundamental variables
image_size = 28          # ‼️ Shouldn't be changed - Read note:          Should only be changed if the dataset is changed to another one with other image dimensions 
output_size = 10         # ‼️ Shouldn't be changed - Read note:          Shouldn't be changed for this dataset. One output node for each number (0-9). If another dataset is used it should match the number of categories and each category (eg. shirt) should be assigned a category number (eg shirt=2)

### Functional variables
reset_network = True     # ✔️  Can be toggled.  Note: it will erase the current training and reset all the weights and biases.
hidden_layers = 2        # ✔️  Can be changed.  There can be any amount of hidden layers. However, if the value is changed the network has to be reset first. That is done by setting the variable "reset_network" to True
hiddenlayer_size = 32    # ✔️  Can be changed.  Note that the size of the hidden layers can currently only be uniform. In other words, its impossible to have the first hidden layer with 32 nodes and the next with 64
learning_rate = 0.05     # ✔️  Can be changed.  It's a very narrow window that works for the network. If changed it's recommended to set plot_graph to True to see the effect it has.
batchsize = 100          # ✔️  Can be changed.  The amount of images the networks sees before changing it's weights. The higher, the more precise the adjustments will be, but will come at the cost of needing a higher learning rate of more epochs
epochs = 10              # ✔️  Can be changed.  The amount of times the images (all 60.000) gets cycled through. Be aware that too many epochs will result in overfitting which makes the network worse on unseen data.

### "Cosmetic" variables - (doesn't have any effects on the neural network itself, but can make the script slower.)
str_len = 30             # ✔️  Can be changed.  The size of the progress bar in the terminal
show_wrong = True        # ✔️  Can be changed.  Shows the predictions that the network gets wrong in the evalutation-dataset
plot_graph = True        # ⚠️  Read note:  Determines if the loss graph gets plotted (if set to true). Be aware that this can make the script up to 4x slower
show_weights = True      # ⚠️  Read note: (network has to be trained first) Is used to see the patterns that the weight-layers have been adjusted to however, it only makes sense when there are 0 hidden layers as that shows patterns that loosely mimics the numbers from 0-9


# ReLU and its derived variant
def ReLU(x):
    return(np.maximum(0,x))

# The derived ReLU function
def ReLU_d(x):
    return(np.where(x > 0, 1,0))

# The softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return(exp_z / np.sum(exp_z))

# Returns the names of the dynamic layers. They are named "weightlayer1", "...2" and so on. (same for bias layers)
def namelist():
    namelist = []
    for i in range(hidden_layers+1):
        namelist.append("weightlayer"+str(i+1))
    for i in range(hidden_layers+1):
        namelist.append("biaslayer"+str(i+1))
    return(namelist)

# Returns the sizes of the different layers.
def layersizes():
    layer_sizes = [0]*(hidden_layers+2)
    layer_sizes[0] = image_size*image_size      # The first layer has the input size (786)
    layer_sizes[-1]= output_size                # The last layer has the output size (10)
    for i in range(hidden_layers):
        layer_sizes[i+1] = hiddenlayer_size     # All the other layers get the size defined by the hiddenlayer_size defined in the start of the script 
    return(layer_sizes)

# returns a list with the sizes of the weight- and biaslayers. Its necessary when using dynamic layers.
def sizelist():
    layer_sizes = layersizes()
    size_list = [0]*(hidden_layers+1)*2
    for i in range(hidden_layers+1):
        size_list[i] = layer_sizes[i]*layer_sizes[i+1]    # the weights  (one for each combination of nodes between this and next layer)
        size_list[i+hidden_layers+1] = layer_sizes[i+1]  # the biases   (one for each node in the layer)
    return(size_list)




# Saves the neural networks layers to be loaded in a new session or to be used as checkpoints.
# Each weightlayer and biaslayer gets saved in a seperate file
def savenetwork(printpath):
    folder_path = "saved_network/"
    for file in os.listdir(folder_path):    
        file_path = os.path.join(currentpath,folder_path, file)
        if printpath == True:
            print("Deleting:",file_path)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file

    for name in namelist():
        layer = globals()[name]
        filesave(name,layer,printpath)

# Saves a numpy array as a file 
# (Each weight- and biaslayer is saved as a numpy array)
def filesave(name,matrix,printpath):
    if printpath == True:
        print('Saving: "saved_network/'+str(name)+'.npy"')

    np.save(currentpath+"/saved_network/"+str(name),matrix)

# Loads all the image data and labels for each element in either the training dataset or the test dataset
def load_images_ubyte(type):
    image_buf = np.fromfile(file=(currentpath+"/"+type+'_images/'+type+'-images.idx3-ubyte'),dtype=np.ubyte)[16:]
    label_buf = np.fromfile(file=(currentpath+"/"+type+'_images/'+type+'-labels.idx1-ubyte'),dtype=np.ubyte)[8:]

    num_images = len(label_buf)    

    imagedata = image_buf.reshape(num_images,image_size,image_size,1)
    rawdata = image_buf.reshape(num_images,image_size*image_size,1)
    rawdata = np.divide(rawdata,255)  # omformer dataen fra 8-bit værdier (altså mellem 0 og 255) til floats mellem 0 og 1
    return(rawdata,imagedata,label_buf,num_images)

# This function shows an image by supplying it with a numpy array containing the imagedata,
# a prediction and the correct label. It can also show weightlayers and the patterns that 
# have been formed in the training process.
def show_image(image_matrix, prediction, correct):
    plt.figure(1)
    plt.clf()

    image = np.asarray(image_matrix.reshape(28,28)).squeeze()
    cmap = mpl.cm.bone      # Best colors are: bone (white-ish on black), gray (white on black) and binary (black on white)

    plt.imshow(image, cmap=cmap)
    title = "Prediction:"+str(prediction)+"   Correct:"+str(correct)
    plt.title(title)
    plt.draw()
    plt.pause(2)

# Evaluates the neural network with the test data.
# This is the actual score of the neural network, and does not necessarily
# correlate with the loss graph. This is a necessary step to ensure that the neural
# network isn't overfitting to the training data. When this score drops it's an indicator
# that the network has hit the overfitting point and its time to stop the training.
def evaluate():
    rawdata, imagedata, labelbuf, num_images = load_images_ubyte("test")
    loss = 0
    for i in range(num_images):
        a1 = rawdata[i]
        forward_propagation(a1)
        target_number = labelbuf[i]
        guess = np.argmax(globals()["a"+str(hidden_layers+2)])

        if guess !=target_number:
            loss +=1
            if show_wrong == True:
                print("Guess:", guess)
                print("Correct answer:", target_number)
                show_image(a1, guess, target_number)

    evaluating_loss = loss/num_images
    classify_percent = round((1-evaluating_loss)*100,3)
    return(evaluating_loss,classify_percent)

# Initializes the layers, while double checking for errors (such as wrong layer sizes)
def initialize_layers():
    name_list = namelist()
    if reset_network == False:
        loaded = 0
        found = []
        missing = []
        for name in name_list:
            try:
                globals()[name] = np.load("saved_network/"+name+".npy")
                loaded +=1
                found.append(name)
                print('Loaded: "saved_network/'+name+'.npy"')
            except:
                print('No array saved at location: "saved_network/'+name+'.npy"')
                missing.append(name)

        # The following tells the user if the saved network has less layers than specified
        if loaded != (hidden_layers+1)*2:
            if loaded%2 == 0:
                print("\nThe saved network has another dimension than",hidden_layers,"hidden layers.")
                print("The following arrays was found:",found)
                print("However, the following arrays was missing:", missing)
                print('Either change "hidden_layser" to the correct size or set "reset_network"=True. Please note the latter will delete and reset all current weights and biases.')
                exit()
            else:
                print('\nThe savedata is missing one or more arrays. Please reset the network by setting "reset_network"=True. Please note that this will delete the rest of the current savedata.')
                exit()

        # The following tells the user if the saved network has more layers than specified
        elif globals()[name_list[-1]].shape[0] != 10:
            print("\nThe saved network has more dimensions than the ",hidden_layers,"hidden layers specified.")
            print("Either change the amount of hidden layers to the correct amount or reset the network.")
            exit()


        # Checks if the hiddenlayer_size is the same as the saved network
        elif hidden_layers>0 and globals()[name_list[-2]].shape[0] != hiddenlayer_size:
            print("The saved network has hidden layers with the size of", globals()[name_list[-2]].shape[0], "neurons per layer.")
            print("The current setting has", hiddenlayer_size,"neurons per layer.")
            print("Either change the hiddenlayer_size variable or set reset_network to True.")
            exit()
            
        else:
            print()

    # Reset the network and initializes the weights randomly and the biases as 0
    elif reset_network == True:
        layer_sizes = layersizes()        
        for i in range(len(layer_sizes)-1):
            globals()[name_list[i]] = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i]) # weights
            globals()[name_list[-i-1]] = np.zeros((layer_sizes[-i-1], 1))            # bias

    else:
        print('\nThe variable "reset_network" has a wrong assigned value. It should either be True or False. Please note that "True" will delete and reset all the current assigned weights and biases.')
        exit()


# Goes forward in the network. 
# If the global-functions are confusing look at the ...simple.py version of the script.
# In that version the functions are written for each layer instead of a generel expression
def forward_propagation(a1):
    a = a1    
    for i in range(hidden_layers+1):
        bias = globals()["biaslayer"+str(i+1)]
        weights = globals()["weightlayer"+str(i+1)]    
        z = np.dot(weights,a)+bias 
        a = ReLU(z)        
        globals()["z"+str(i+2)] = z
        globals()["a"+str(i+2)] = a
    
    # the last layer doesnt use ReLU so overwrites with softmax activation instead
    globals()["a"+str(hidden_layers+2)] = softmax(z)

    return

# Resets all the changes in weights to zero as they have been applied.
def reset_weights():    
    layer_sizes = layersizes()
    for i in range(hidden_layers+1):        
        globals()["w"+str(i+1)] = np.resize([[float(0)]],(layer_sizes[i+1],layer_sizes[i]))
        globals()["b"+str(i+1)] = np.resize([[float(0)]],(layer_sizes[i+1],1))


# Takes the intended changes in the weights and biases and applies them to the weight and bias layers
def apply_backpropagation():
    for i in range(hidden_layers+1):
        globals()["weightlayer"+str(i+1)] += globals()["w"+str(i+1)]*learning_rate/batchsize 
        globals()["biaslayer"+str(i+1)] += globals()["b"+str(i+1)]*learning_rate/batchsize
    reset_weights()
    return
    

# Again: if the global-functions are confusing look at the ...simple.py version of the script.
# In that version the functions are written for each layer instead of a generel expression
# 
# The backpropagation function is created to accomodate the dynamic layers
# It finds the derivative of the deviation of the intended output and targeted 
# output to find the gradient and takes a step in the direction of steepest descent. 
# The step size is determined by the learning rate. 
# (however, the learningrate gets multiplied in the apply_backpropagation function to 
# eliminate batchsize-1 multiplications for each batchsize)
def backpropagation(targetlist,a1):
    globals()["dz"+str(hidden_layers+2)] = targetlist - globals()["a"+str(hidden_layers+2)]
    
    for i in range(hidden_layers):
        dz = globals()["dz"+str(hidden_layers+2-i)]
        ind = str(hidden_layers+1-i)
        globals()["db"+ind] = dz
        globals()["dw"+ind] = np.dot(dz,(globals()["a"+ind]).transpose())
        globals()["dz"+ind] = np.dot(globals()["weightlayer"+ind].transpose(),dz)*ReLU_d(globals()["z"+ind])
            
    globals()["db1"] = globals()["dz2"]
    globals()["dw1"] = np.dot(globals()["dz2"],a1.transpose())

    return


# creates the empty plot for the loss graph
def create_loss_plot():
    global fig, ax, x_data, y_data, line
    plt.ion()
    plt.figure(2)

    fig, ax = plt.subplots(num=2)
    x_data, y_data = [], []
    (line,) = ax.plot([], [], "bo")

    ax.set_ylim(0,1)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Loss graph (lower is better)")

# adds a point to the loss graph
def add_point(x, y):
    x_data.append(x)
    y_data.append(y)
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.relim()  
    ax.autoscale_view()  
    fig.canvas.draw()  
    fig.canvas.flush_events()




def train():
    loss = 0
    percentage = 0

    # self explanatory
    initialize_layers()
    # initilizes all the weight and bias changes as 0 (so the vars exist)
    reset_weights()

    # Determines whether the loss_graph gets created or not
    if plot_graph == True:
        print("The loss graph is turned on (plot_graph=True)." \
        "\nIt's great for getting an understanding of the training progress." \
        "\nHowever, be aware that it makes the script up to 4x slower.\n")
        
        create_loss_plot()        
    
    # loads all the training images and labels
    rawdata, image_array, labels, num_images = load_images_ubyte("train")

    
    # The script simply loops through. It can be stopped after each iteration as there are 
    # checkpoints after each epoch
    for epoch in range(1,epochs+1):
        epoch_start = time.time()

        # "Splits" the data up so it knows how many times it can loop 
        # through the dataset with the given batchsize    
        for x in range(num_images//batchsize):
            # the following 4 lines is just the updating progressbar for each epoch
            percent_ = (x+1)*batchsize/num_images
            fractions = round(percent_*str_len)
            expected_remaining = (time.time()-epoch_start)*(1/percent_-1)
            print(f"Epoch {epoch}/{epochs}:\t[{fractions*'#'+(str_len-fractions)*'.'}] {round(percent_*100,1)}%\t  Runtime: {round(time.time()-epoch_start,2)}s  \tExpected time remaining time: {round(expected_remaining,1)}s     ", end='\r')

            # Then loops through each item in the given batch
            for item in range(batchsize):
                a1 = rawdata[x*batchsize+item]
                
                # First doing forward propagation
                prediction_ = forward_propagation(a1)
                target_number = labels[x*batchsize+item]
                guess = np.argmax(globals()["a"+str(hidden_layers+2)])
                if guess != target_number:
                    loss +=1            
                targetarray = np.resize(0, (10,1))
                targetarray[target_number] = 1

                # And then backwardspropagation where it saves the changes that should 
                # be made to the weights and biases
                backpropagation(targetarray,a1)            
                for layer in range(hidden_layers+1):
                    ind = str(layer+1)
                    globals()["w"+ind] += globals()["dw"+ind]
                    globals()["b"+ind] += globals()["db"+ind]


            # After each batch the sum of weight-changes gets applied
            apply_backpropagation()
            # and the loss in the batch gets added to the plot if its turned on
            if plot_graph == True:
                add_point((x+1)*(batchsize),loss/batchsize)
                loss = 0
        
        # Saves the network as a checkpoint so that in the case scritps gets killed, 
        # the only progress lost is just the current unfnished epoch
        savenetwork(printpath=False)
        
        # Evaluate the neural network against the evaluation dataset to check its actual performance
        prev_percentage = percentage
        loss_, percentage = evaluate()

        # Prints the training time for the epoch and the classifying percentage for the neural network    
        print(f"Epoch {epoch}/\t[{fractions*'#'+(str_len-fractions)*'.'}] {round(percent_*100,1)}%\t  Runtime: {round(time.time()-epoch_start,2)}s  \tClassifying percentage (on test data): {percentage}%     \tNetwork saved✔️{' '*50}")
        
        # If the neural networks begins to perform worse on the evaluation dataset
        # it's an indication that the script is overfitting to the training data
        # That means the neural network is beginning to remember the dataset 
        # instead of learning patterns in the data. 
        # There is no workaround for this phenomenon other than stopping training
        if prev_percentage>percentage:
            warning_string = "⚠️   The classification percentage has dropped. This could be a sign of overfitting. It's recommended to stop the training if it continues dropping."
            print(warning_string)

    # Shows the patterns in the weightlayers. 
    # However, this only makes sense if there are no hidden layers as 
    # that will show patterns resembling the actual numbers from 0 to 9
    if show_weights == True:
        for i in range(layersizes()[1]):
            show_image(globals()["weightlayer1"][i],"",i)

# needed to be able to find the path of subfiles
currentpath = os.path.dirname(os.path.abspath(__file__))
train()
