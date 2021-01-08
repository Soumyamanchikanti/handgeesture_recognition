from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import cv2

model= Sequential()

#add conv layers and pooling layers 
model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5)) 

model.add(Flatten())


commandmodel.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.5))

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))


model.add(Dropout(0.5))

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

#output layer
model.add(Dense(output_dim = 4, activation = 'sigmoid'))


#Now copile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory("Dataset/training_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')
test_set=test_datagen.flow_from_directory("Dataset/test_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')
model.fit_generator(training_set,
                         samples_per_epoch = 1000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 320)
model.save_weights("weights.hdf5",overwrite=True)
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")


img = load_img('Dataset/test_set/b/b26.jpg',target_size=(200,200))
x=array(img)
img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
img=img.reshape((1,)+img.shape)
img=img.reshape(img.shape+(1,))

test_datagen = ImageDataGenerator(rescale=1./255)
m=test_datagen.flow(img,batch_size=1)
y_pred=model.predict_generator(m,1)
plot_model(model, to_file='model.png', show_shapes = True)

