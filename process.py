import os
import cv2
import glob
import csv
import copy
import time
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from cnn1 import cnn_retina

YROW = [0, 0, 0, 0, 0]  # initialized row for output (1 entered for classification)
seed = 21               # Set random seed for purposes of reproducibility
ylim = [0, 1.5]

def plot(epochs, history):
    xdata = list(range(1, epochs + 1))
    trace1 = go.Scatter(
                    x = xdata,
                    y = history['acc'],
                    name='Training Accuracy',
                    line=dict(color='green'))

    trace2 = go.Scatter(
                    x = xdata,
                    y = history['val_acc'],
                    name='Validation Accuracy',
                    line=dict(color='blue'))

    trace3 = go.Scatter(
                    x = xdata,
                    y = history['loss'],
                    name='Training Loss',
                    line=dict(color='red'))

    trace4 = go.Scatter(
                    x = xdata,
                    y = history['val_loss'],
                    name='Validation_Loss',
                    line=dict(color='orange'))

    layout = go.Layout(
        showlegend=True,
        xaxis = dict(
            title='Epoch'),
        yaxis=dict(
            title='Value',
            range=ylim,
            ))

    # can show manual ticks with tickvals = []

    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data,layout=layout)

    plotly.offline.plot(fig, filename='TextClass.html', auto_open=True)

def main():

    # initialize timer
    start_time = time.process_time()

    # read in csv file with images and classifications
    with open('D:\\Data\\Retinop\\train.csv', newline='') as File:  
        reader = csv.reader(File)
        
        idx = 0
        rdict = {}
        for row in reader:
            if idx != 0: 
                rdict[row[0]] = int(row[1])
            idx += 1
        print("Number of Retina Images in train.csv:", idx, len(rdict))

        X = []
        y = []

    # read in preprocessed images and make X, y arrays 
    retinopFiles = glob.glob('D:\\Data\\Retinop\\PreProcessed\\train_images\\*.png')
    print("Number of Retina Images: ", len(retinopFiles))
    for retina in retinopFiles:
        img = cv2.imread(retina)
        X.append(img)
        rname = os.path.basename(retina).split('.')[0]
        yrow = copy.deepcopy(YROW)
        if rname in rdict:
            idx = rdict[rname]
            yrow[idx] = 1
            y.append(yrow)

    print("Data lengths: X - {}  y - {}".format(len(X), len(y)))
    print('READ DATA FROM FILES... elapsed time: {} sec'.format(time.process_time() - start_time))

    X = np.array(X)
    y = np.array(y)
    # print(X.shape, y.shape)
    # Split data for training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize Keras CNN model (images are 320 x 320 x 3 with 5 classifications)
    epochs = 64
    model = cnn_retina(X_train.shape[1:], 5, dropout=0.5, reg=0.01)
    history = model.fit(X_train, y_train, batch_size=10, epochs=epochs, validation_data=(X_test, y_test))
    print('TRAINING AND VALIDATION COMPLETE... elapsed time: {} sec'.format(time.process_time() - start_time))

    model.save('retinopathy1.h5')
    plot(epochs, history.history)


if __name__ == "__main__":
    main()