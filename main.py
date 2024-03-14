import pandas as pd
import tkinter as tk
from tkinter import*
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
main = tk.Tk()
main.title("MACHINE LEARNING MODEL FOR PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
main.geometry("1600x1300")
title = tk.Label(main, text="MACHINE LEARNING MODEL FOR PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES",justify='center')

def upload():
    global filename
    global dataset,categories
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    path = r"dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    categories
    text.insert(END,"Total Categories Found In Dataset"+str(categories)+'\n\n')
def imageprocessing():
    global flat_data,target
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir=r"Dataset"
    #create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')

    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        text.insert(END,"Total Images Found In Dataset : "+str(flat_data.shape[0])+'\n\n')
        
    else:
        #path which contains all the categories of images
        for i in categories:
            print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            #create file paths by combining the datadir (data directory) with the i
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))#Reads the image using imread.
                img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
                flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
                target_arr.append(categories.index(i)) #Adds the index of the category to the target_arr.
                #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
                print(f'loaded category:{i} successfully')
                #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
def splitting():
    global x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.20,random_state=77)
    text.insert(END,"Total Images Used For Training : "+str(x_train.shape[0])+'\n\n')
    text.insert(END,"Total Images Used For Testing : "+str(x_test.shape[0])+'\n\n')
def naivebayes():
    from sklearn.naive_bayes import GaussianNB
    # Initializing and training the Gaussian Naive Bayes model
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred1 = nb_classifier.predict(x_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred1)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Generating a classification report
    report = classification_report(y_test, y_pred1)
    print("\nNaive Bayes model classification_report:\n",report)
    text.insert(END,"Naivebayes Accuracy : "+str(accuracy*100)+'\n\n')
    text.insert(END,"Naivebayes Classification Report: "+'\n'+str(report)+'\n\n')
    cm=confusion_matrix(y_test,y_pred1)
    class_labels=['pneumonia','Normal']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Naive Bayes model Confusion Matrix")
    plt.show()
    
def RFC():
    global classifier
    text.delete('1.0', END)
    filename = 'Classifier.pkl'
    if os.path.exists(filename):
        # Load the trained model from the Pickle file
        with open(filename, 'rb') as model_pkl:
            classifier = pickle.load(model_pkl)
    else:
        # Create and train the Random Forest Classifier
        classifier = RandomForestClassifier()   
        classifier.fit(x_train, y_train)
        # Save the trained model to a Pickle file
        with open(filename, 'wb') as model_pkl:
            pickle.dump(classifier, model_pkl)
            # Evaluate the model
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("Accuracy:", acc)
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    text.insert(END,"RFC Accuracy : "+str(acc)+'\n\n')
    text.insert(END,"RFC Classification Report: "+'\n'+str(report)+'\n\n')
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_labels = ['Pneumonia','NORMAL']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("RFC classifier Confusion Matrix")
    plt.show()
        

def prediction():
    path = filedialog.askopenfilename(initialdir = "testing")
    img=imread(path)
    img_resize=resize(img,(150,150,3))
    img_preprocessed=[img_resize.flatten()]
    output_number=classifier.predict(img_preprocessed)[0]
    output_name=categories[output_number]

    plt.imshow(img)
    plt.text(10, 10, f'Predicted Output: {output_name}', color='white',fontsize=12,weight='bold',backgroundcolor='black')
    plt.axis('off')
    plt.show()
    

title.grid(column=0, row=0)
font=('times', 15, 'bold')
title.config(bg='purple', fg='white')
title.config(font=font)
title.config(height=3,width=120)
title.place(x=60,y=5)

uploadButton = Button(main, text="Upload Dataset   ",command=upload)
uploadButton.config(bg='Skyblue', fg='Black')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Image Processing ",command=imageprocessing)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=250,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Splitting   ",command=splitting)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=450,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Naive Bayes   ",command=naivebayes)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=600,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="RFC Classifier ",command=RFC)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=770,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Prediction   ",command=prediction)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=950,y=100)
uploadButton.config(font=font)

font1 = ('times', 12, 'bold')
text=Text(main,height=28,width=180)

scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=15,y=250)
text.config(font=font1)


main.mainloop()