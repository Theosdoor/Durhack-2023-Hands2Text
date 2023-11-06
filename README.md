# Hands-2-Text (Durhack Project 2023)

## I Introduction  
For our 24-hour hack we created a convolutional neural network (CNN) model (goated.keras & goated2.keras) that takes images of ASL signs for letters of the alphabet, and returns the letter represented (see `Durhack23.ipynb` for the code). We developed a web interface for this, where users could take photos of their signs to communicate more easily with others online. However, unfortunately we found it surprisingly difficult to program the ability to submit a photo through the website (via the webcam or otherwise) which could then be predicted by our model, and we didn't manage to complete this in time. Although, this would be a great improvement if we tried such a project again. 

Our CNN model was developed upon [this example](https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy/notebook). We trained it on 50 x 50 pixel images with greyscale values between 0-255. goated.keras was our original model, with 20 epochs of training. goated2.keras underwent 30 epochs.

Please read the Log.txt file to see a brief diary of our 24-hours!

## II How to Use the CNN Model (without the website)

**Input requirements:** Single 50 x 50 pixel image with greyscale values between 0-255. It must be a .jpg file. If it is a different format there are plenty of free conversion tools available, such as [Cloud Convert](https://cloudconvert.com/).
**Prior to using the model:** Download either goated.keras or goated2.keras and install tensorflow using pip. Then create a new python program.
**Using the model:** First, load the model:
```
import tensorflow as tf
model = tf.keras.models.load_model("filepath/to/goated.keras") # change to goated2.keras to try the other model.
model.summary() # shows layers in model
```   
This is further outlined on [this page](https://www.tensorflow.org/guide/keras/serialization_and_saving). To input an image, we must resize it in order to be accepted by the model. The code below shows how to process an image, `img.jpg`, and how to use our model to predict the letter it is signing:
```
image_path = 'path/to/img.jpg'
image_size = [50,50]

# Convert image to tensor
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels = 1) # This greyscales the image as well
image = tf.image.resize(image, image_size)

# Convert to numpy array
data = image.numpy() / 255 # to normalise the data
pred = model.predict(data.reshape(-1, 50, 50, 1)) # reshape to be accepted by model
pred_arr = np.array(pred[0])

# Link to letter predictions, and print 3 most likely letters according to the model.

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing']

letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
predarrayordered = sorted(predarray, reverse=True)
high1 = predarrayordered[0]
high2 = predarrayordered[1]
high3 = predarrayordered[2]
for key,value in letter_prediction_dict.items():
    if value==high1:
        print("Predicted Character 1: ", key)
        print('Confidence 1: ', 100*value)
    elif value==high2:
        print("Predicted Character 2: ", key)
        print('Confidence 2: ', 100*value)
    elif value==high3:
        print("Predicted Character 3: ", key)
        print('Confidence 3: ', 100*value)
```
This will print the 3 most likely letters according the the model, along with the model's percentage confidence in each prediction.

## III Project Evaluation



## IV Tools/resources Used

- **Tensorflow and keras:** to design, train and use the CNN model.
- **Google Cloud:** to run the model training, which was a lot (~40%) faster than using our own machines.
- **Kaggle:** for the ASL datasets and tutorials/learning materials for CNNs.

### Important Links and References

- [Inspiration and starting point for our CNN](https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy/notebook#Loading-the-ASL-dataset)
- [A supplement to the above](https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442)
- [The first dataset we used](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [The second (and larger) dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)
- [CNN Kaggle Tutorial](https://www.kaggle.com/code/ryanholbrook/the-convolutional-classifier)
