import logging
import platform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from flask import Flask, render_template, request
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import backend as K
# import keras
# import matplotlib.pyplot as plt
# from keras.models import model_from_json
# from keras import backend as K
# from keras.utils import plot_model
# from keras.preprocessing.image import ImageDataGenerator
# from PIL import Image
# import numpy as np
# from _functools import reduce
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from numpy import expand_dims
# from keras.models import load_model

# import pathlib
# from keras.models import Model
# import matplotlib.gridspec as gridspec

app = Flask(__name__)



@app.route('/')
def home():
    return render_template("index.html")
upload_graph='static/graph/'
app.config['upload_graph'] = upload_graph
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
@app.route('/train',methods = ['POST', 'GET'])
def train():
    if request.method == 'GET':
        return render_template("train.html")
    else:
        K.clear_session()
        #function to get dimensions of images in training directory
        def imgsize(train_dir):
            global img_width
            global img_height
            for subdir, dirs, files in os.walk(train_dir):
                for file in files:
                    for x in range(1):
                        img= os.path.join(subdir, file)
                        im = Image.open(img)
                        img_width, img_height = im.size
                        return img_width, img_height
        #input data from html frontend
        epochs = int(request.form.get('epochs'))
        batch_size = int(request.form.get('batch_size'))
        optimizer = str(request.form.get('optimizer'))
        model_name=str(request.form.get('model_name'))
        train_dir = str(request.form.get('train_dir'))
        val_dir = str(request.form.get('val_dir'))
        conv_layers = int(request.form.get('conv_layers'))
        pool_layers = int(request.form.get('pool_layers'))
        dense_layers = int(request.form.get('dense_layers'))
        classes = int(len(next(os.walk(train_dir))[1]))
        nb_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
        nb_validation_samples = sum([len(files) for r, d, files in os.walk(val_dir)])
        print(pool_layers)
        imgsize(train_dir)
        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 1)
        #function to add convolution layers
        def conv(layers):
            for layer in range(layers):
                filters = int(request.form.get("conv_filter"+str(layer+1)))
                size = int(request.form.get("conv_size"+str(layer+1)))
                model.add(Conv2D(filters, kernel_size=(size,size),
                                 activation=str(request.form.get('conv_activation')),
                                 input_shape=input_shape))
        #function to add pooling layers
        def pool(layers):
            for layer in range(layers):
                type = str(request.form.get("pool_type"+str(layer+1)))
                size = int(request.form.get("pool_size"+str(layer+1)))
                if type == "MaxPooling2D":
                    model.add(MaxPooling2D(pool_size=(size,size)))
                else:
                    model.add(AveragePooling2D(pool_size=(size,size)))
        #function to add dense layers
        def dense(layers):
            for layers in range(layers):
                nodes = int(request.form.get("dense_nodes"+str(layers+1)))
                model.add(Dense(nodes, activation= str(request.form.get('dense_activation'+str(layers+1)) )))
        #model layers
        model = Sequential()
        conv(conv_layers)
        pool(pool_layers)
        model.add(Flatten())
        #model.add(Dense(300, activation='relu'))
        dense(dense_layers)
        model.add(Dropout(0.3))
        model.add(Dense(classes))
        #adding activation function based on input from html
        model.add(Activation(str(request.form.get('class_activation'))))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer ,
                      metrics=['categorical_accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                rescale=1./255)

        # this is the augmentation configuration we will use for validation:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        #compiling training set
        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(img_width, img_height),
                color_mode = 'grayscale',
                batch_size= batch_size,
                shuffle= 'false',
                class_mode='categorical')
        #compiling validation set
        validation_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=(img_width, img_height),
                color_mode = 'grayscale',
                batch_size=8,
                shuffle = 'false',
                class_mode='categorical')
        #fitting the model
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)
        model.save(model_name +'.h5') #output model
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        #evaluating model on validation set
        score=model.evaluate_generator(generator=validation_generator,steps=nb_validation_samples)
        #plotting model architecture
        plot_model(model, to_file=upload_graph+'model.png',show_shapes=True)
        loss = str(round(score[0],2))
        acc = str(round(score[1]*100,2))+'%'

        print('Validation loss:', score[0])
        print('Validation:', score[1])
        train= 'Network trained and saved as '+model_name+'.h5'
        #plotting training graph
        plt.subplot(211)
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # summarize history for loss
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.suptitle(optimizer+'Batch Size'+str(batch_size), size=16)
        plt.subplots_adjust(hspace=.5)
        plt.savefig(upload_graph+'graph.png')
        graph = upload_graph+'graph.png'
        archi = upload_graph+'model.png'
        graph_text = 'Training_History_Graph'
        archi_text = 'Model_Structure'

        K.clear_session()

        #uploading necessary details back to html
        return render_template("train.html",train_dir=train_dir,val_dir=val_dir,
        epochs=epochs,batch_size=batch_size,optimizer=optimizer,train=train,
        graph=graph,archi=archi,graph_text=graph_text,archi_text=archi_text,loss=loss,acc=acc)
@app.route('/test',methods = ['POST', 'GET'])
def test():
    if request.method == 'GET':
        return render_template("test.html")
    else:
        K.clear_session()
        test_dir = str(request.form.get('test_dir'))
        model_dir = str(request.form.get('model_dir'))

        model = load_model(model_dir) #input trained model
        for layer in model.layers:
            global img_size
            img_size = layer.input_shape[1]
            break
        x = 0
        for subdir, dirs, files in os.walk(test_dir):
            for file in files:
                x = x+1
        print(x)

        testimages = x #number of images to test

        datagen = ImageDataGenerator(rescale=1./255) #change values to 0 and 1
        generator = datagen.flow_from_directory(
                test_dir,
                target_size=(img_size, img_size), #dimensions of test image
                color_mode="grayscale",
                shuffle = False,
                class_mode='categorical',
                batch_size=1)
        #accuracy of all images
        score=model.evaluate_generator(generator=generator,steps=testimages)
        x = 0
        for name in model.metrics_names:
            if name == 'loss':
                test_loss = str(round(score[x],2))
            if name == 'categorical_accuracy':
                test_acc = str(round(score[x]*100,2))+'%'
            x+=1
        K.clear_session()
        return render_template("test.html",test_loss=test_loss,test_acc=test_acc,test_dir=test_dir,model_dir=model_dir,img_size=str(img_size),img_num=str(testimages))

upload_image='static/image/'
upload_filters = 'static/filters/'
app.config['upload_image'] = upload_image
@app.route('/testsingle',methods = ['POST', 'GET'])
def testsingle():
    if request.method == 'GET':
        return render_template("testsingle.html")
    else:
        K.clear_session()
        image = request.files['image']
        image.save(upload_image+image.filename)

        # img.show()
        model_dir = str(request.form.get('model_dir'))
        categ_dir = str(request.form.get('categ_dir'))


        model = load_model(model_dir)
        for layer in model.layers:
            global x
            x = layer.input_shape[1]
            break
        img = load_img(upload_image+image.filename,target_size=(x, x),color_mode = "grayscale")
        img = img_to_array(img)
        img = expand_dims(img, axis=0)


        # Predicting the Test set results

        img /= 255
        predict = model.predict(img)
        print(x)
        print(predict)
        datagen = ImageDataGenerator(rescale=1./255)

        generator = datagen.flow_from_directory(
                categ_dir)

        # filenames = test_generator.filenames
        # nb_samples = len(filenames)
        # generator.reset()
        # # predict = model.predict_generator(generator,steps = testimages)
        index = np.argmax(predict,axis=1)


        labels = (generator.class_indices)

        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in index]
        print(predictions[0])
        predict_url = upload_image+predictions[0]+'.png'
        test_url = upload_image+image.filename
        print(test_url)
        filters_urlArr = []
        filters_name= []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            # Check for convolutional layer
            # print(layer.name)
            if 'conv' not in layer.name:
                if 'max' not in layer.name:
                    if 'average' not in layer.name:
                        continue
            # Summarize output shape
            print(layer.output.shape[3])
            modelInfo = Model(inputs=model.inputs, outputs=model.layers[i].output)
            # prepare the image (e.g. scale pixel values for the vgg)

            # get feature map for first hidden layer
            feature_maps = modelInfo.predict(img)
            # print(feature_maps)
            # plot all 64 maps in an 8x8 squares
            # print(img.shape)
            count = 0
            number = int(layer.output.shape[3])
            numArr = []
            for i in range(2, number-1):
                if number%i == 0:
                    numArr.append(i)
                    i += 1;
                    count += 1
            size = int(len(numArr))
            if size % 2 == 0:
                int1 = int(size/2 - 1)
                int2 = int(size/2)
                num1 = numArr[int1]
                num2 = numArr[int2]
            else:
                int1 = int(size/2)
                num1 = numArr[int1]
                num2 = numArr[int1]
                print(num1)
            plt.figure(figsize = (num1,num2))
            plt.gcf().set_size_inches(20, 10)
            gs1 = gridspec.GridSpec(num1,num2)
            gs1.update(wspace=0.02, hspace=0.01)
            for ix in range(number):
                # specify subplot and turn of axis
                ax = plt.subplot(gs1[ix])
                plt.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_aspect('equal')
                plt.subplots_adjust(wspace=None, hspace=None)
            # plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix],cmap='gray')
            # show the figure
            plt.savefig( upload_filters+layer.name+image.filename,bbox_inches='tight')
            filter_url = upload_filters+layer.name+image.filename
            filters_urlArr.append(filter_url)
            filters_name.append(layer.name)
        simulfilters_urlArr =[]
        img2 = load_img(upload_image+predictions[0]+'.png',target_size=(x, x),color_mode = "grayscale")
        img2 = img_to_array(img2)
        img2 = expand_dims(img2, axis=0)
        img2 /= 255
        for i in range(len(model.layers)):
            layer = model.layers[i]
            # Check for convolutional layer
            # print(layer.name)
            if 'conv' not in layer.name:
                if 'max' not in layer.name:
                    if 'average' not in layer.name:
                        continue
            # Summarize output shape
            print(layer.output.shape[3])
            modelInfo = Model(inputs=model.inputs, outputs=model.layers[i].output)
            # prepare the image (e.g. scale pixel values for the vgg)

            # get feature map for first hidden layer
            feature_maps = modelInfo.predict(img2)
            # print(feature_maps)
            # plot all 64 maps in an 8x8 squares
            # print(img.shape)
            count = 0
            number = int(layer.output.shape[3])
            numArr = []
            for i in range(2, number-1):
                if number%i == 0:
                    numArr.append(i)
                    i += 1;
                    count += 1;
            size = int(len(numArr))
            if size % 2 == 0:
                int1 = int(size/2 - 1)
                int2 = int(size/2)
                num1 = numArr[int1]
                num2 = numArr[int2]
            else:
                int1 = int(size/2)
                num1 = numArr[int1]
                num2 = numArr[int1]
                print(num1)
            plt.figure(figsize = (num1,num2))
            plt.gcf().set_size_inches(20, 10)
            gs1 = gridspec.GridSpec(num1,num2)
            gs1.update(wspace=0.02, hspace=0.01)
            for ix in range(number):
                # specify subplot and turn of axis
                ax = plt.subplot(gs1[ix])
                plt.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_aspect('equal')
                plt.subplots_adjust(wspace=None, hspace=None)
            # plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix],cmap='gray')
            # show the figure
            plt.savefig( upload_filters+layer.name+predictions[0]+'.png',bbox_inches='tight')
            filter_url = upload_filters+layer.name+predictions[0]+'.png'
            simulfilters_urlArr.append(filter_url)

        # Predicting the Test set results
        K.clear_session()
        return render_template("testsingle.html",simulfilters_names = zip(simulfilters_urlArr,filters_name),filters_names = zip(filters_urlArr,filters_name), model_dir = model_dir,categ_dir=categ_dir,result = predictions[0],predict_url=predict_url,test_url=test_url)


if __name__ == '__main__':
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
