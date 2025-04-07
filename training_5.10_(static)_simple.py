import numpy as np                  # used for the the matrix datastructure
import matplotlib.pyplot as plt     # used for plotting the graphs
import time                         # used for tracking the training time
import os

### Fundamental variables
image_size = 28         # ❗ Shouldn't be changed - Read note:          Should only be changed if the dataset is changed to another one with other image dimensions 
output_size = 10        # ❗ Shouldn't be changed - Read note:          Shouldn't be changed for this dataset. One output node for each number (0-9). If another dataset is used it should match the number of categories and each category (eg. shirt) should be assigned a category number (eg shirt=2)
### Functional variables
learning_rate = 0.05    # ✔️  Can be toggled
batchsize = 200         # ✔️  Can be toggled
epochs = 10             # ✔️  Can be toggled
### "Cosmetic" variables
plot_graph = True       # ✔️  Can be toggled
str_len = 30            # ✔️  Can be toggled        determines the size of the progressbar in the terminal

def ReLU(x):
    return(np.maximum(0,x))

def ReLU_dif(x):
    return(np.where(x > 0, 1,0))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def load_images_ubyte(type):
    image_buf = np.fromfile(file=currentpath+'/'+type+'_images/'+type+'-images.idx3-ubyte',dtype=np.ubyte)[16:]
    label_buf = np.fromfile(file=currentpath+'/'+type+'_images/'+type+'-labels.idx1-ubyte',dtype=np.ubyte)[8:]

    num_images = len(label_buf)    

    imagedata = image_buf.reshape(num_images,image_size,image_size,1)
    rawdata = image_buf.reshape(num_images,image_size*image_size,1)
    rawdata = np.divide(rawdata,255)  
    
    return(rawdata,imagedata,label_buf,num_images)

def evaluate(a1,w1,w2,w3,b1,b2,b3):
    rawdata, imagedata, labelbuf, num_images = load_images_ubyte("test")
    loss = 0
    for i in range(num_images):
        a1 = rawdata[i]
        a1,z2,a2,z3,a3,z4,a4 = forward_propagation(a1,w1,w2,w3,b1,b2,b3)
        target_number = labelbuf[i]
        guess = np.argmax(a4)
        if guess !=target_number:
            loss +=1
    evaluating_loss = loss/num_images
    classify_percent = round((1-evaluating_loss)*100,3)
    return(evaluating_loss,classify_percent)

def initialize_layers():
    w1 = np.random.randn(32, 784) * np.sqrt(2 / 784) 
    w2 = np.random.randn(32, 32) * np.sqrt(2 / 32) 
    w3 = np.random.randn(10, 32) * np.sqrt(2 / 32)
    b1 = np.zeros((32, 1))            
    b2 = np.zeros((32, 1)) 
    b3 = np.zeros((10, 1))
    return(w1,w2,w3,b1,b2,b3)
    
def forward_propagation(a1,w1,w2,w3,b1,b2,b3):
    z2 = w1@a1+b1
    a2 = ReLU(z2) 

    z3 = w2@a2+b2
    a3 = ReLU(z3) 

    z4 = w3@a3+b3
    a4 = softmax(z4)

    return(a1,z2,a2,z3,a3,z4,a4)

def apply_backpropagation(w1,w2,w3,b1,b2,b3,db1,db2,db3,dw1,dw2,dw3):
    w1 += dw1*(learning_rate/batchsize)
    w2 += dw2*(learning_rate/batchsize)
    w3 += dw3*(learning_rate/batchsize)
    b1 += db1*(learning_rate/batchsize)
    b2 += db2*(learning_rate/batchsize)
    b3 += db3*(learning_rate/batchsize)    
    return(w1,w2,w3,b1,b2,b3)

def backpropagation(targetlist,a1,z2,a2,z3,a3,z4,a4,w1,w2,w3):
    dz4 = targetlist - a4

    db3 = dz4
    dw3 = dz4@a3.T
    dz3 = w3.T@dz4*ReLU_dif(z3)

    db2 = dz3
    dw2 = dz3@a2.T
    dz2 = w2.T@dz3*ReLU_dif(z2)

    db1 = dz2
    dw1 = dz2@a1.T
    return(db1,db2,db3,dw1,dw2,dw3)

def create_loss_plot():
    global fig, ax, x_data, y_data, line

    plt.ion()
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    (line,) = ax.plot([], [], "bo")

    ax.set_ylim(0,1)
    ax.set_xlabel("Images")
    ax.set_ylabel("Loss")
    ax.set_title("Loss graph (lower is better)")

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
    w1,w2,w3,b1,b2,b3 = initialize_layers()
    rawdata, image_array, labels, num_images = load_images_ubyte("train")

    if plot_graph == True:
        print("The loss graph is turned on (plot_graph=True)." \
        "\nIt's great for getting an understanding of the training progress." \
        "\nHowever, be aware that it makes the script up to 4x slower.")
        create_loss_plot()    

    for epoch in range(1, 1+epochs):
        epoch_start = time.time()
        for x in range(num_images//batchsize):
            percent_ = (x+1)*batchsize/num_images
            fractions = round(percent_*str_len)
            expected_remaining = (time.time()-epoch_start)*(1/percent_-1)
            print(f"Epoch {epoch}/{epochs}:\t[{fractions*'#'+(str_len-fractions)*'.'}] {round(percent_*100,1)}%\t  Runtime: {round(time.time()-epoch_start,2)}s  \tExpected time remaining time: {round(expected_remaining,1)}s     ", end='\r')

            db1,db2,db3,dw1,dw2,dw3 = [0,0,0,0,0,0]
            for i in range(batchsize):
                a1 = rawdata[x*batchsize+i]
                a1,z2,a2,z3,a3,z4,a4 = forward_propagation(a1,w1,w2,w3,b1,b2,b3)

                target_number = labels[x*batchsize+i]
                guess = np.argmax(a4)
                if guess != target_number:
                    loss +=1
                
                targetarray = np.resize(0, (10,1))
                targetarray[target_number] = 1

                db1_,db2_,db3_,dw1_,dw2_,dw3_ = list(backpropagation(targetarray,a1,z2,a2,z3,a3,z4,a4,w1,w2,w3))
                db1,db2,db3,dw1,dw2,dw3 = db1+db1_,db2+db2_,db3+db3_,dw1+dw1_,dw2+dw2_,dw3+dw3_
                
            w1,w2,w3,b1,b2,b3 = apply_backpropagation(w1,w2,w3,b1,b2,b3,db1,db2,db3,dw1,dw2,dw3)
            if plot_graph == True:        
                add_point((x+1)*(batchsize),loss/batchsize)
                loss = 0
        
        loss_, percentage = evaluate(a1,w1,w2,w3,b1,b2,b3)
        print(f"Epoch {epoch}/\t[{fractions*'#'+(str_len-fractions)*'.'}] {round(percent_*100,1)}%\t  Runtime: {round(time.time()-epoch_start,2)}s  \tClassifying percentage (on test data): {percentage}%{' '*50}")


currentpath = os.path.dirname(os.path.abspath(__file__))
train()
plt.ioff()
plt.show()
