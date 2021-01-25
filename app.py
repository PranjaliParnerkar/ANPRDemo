import skimage
from skimage import io
from flask import Flask, render_template, send_from_directory, request, abort
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images'

app = Flask(__name__)

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
app.config['UPLOAD_EXTENSIONS'] = ['png', 'jpg', 'jpeg', 'gif']
model = load_model('anpr_model.h5')


def fix_dimension(img):
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img


def find_contours(dimensions, img):
        # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]

        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        ii = cv2.imread('contour.jpg')

        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs:
                # detects contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

                # checking the dimensions of the contour to filter out the characters by contour's size
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
                        x_cntr_list.append(
                                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

                        char_copy = np.zeros((44, 24))
                        # extracting each character using the enclosing rectangle's coordinates.
                        char = img[intY:intY + intHeight, intX:intX + intWidth]
                        char = cv2.resize(char, (20, 40))

                        cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)

                        #             Make result formatted for classification: invert colors
                        char = cv2.subtract(255, char)

                        # Resize the image to 24x44 with black border
                        char_copy[2:42, 2:22] = char
                        char_copy[0:2, :] = 0
                        char_copy[:, 0:2] = 0
                        char_copy[42:44, :] = 0
                        char_copy[:, 22:24] = 0

                        img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

        # Return characters on ascending order with respect to the x-coordinate (most-left character first)

        # arbitrary function that stores sorted list of character indeces
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])  # stores character images according to their index
        img_res = np.array(img_res_copy)

        return img_res


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('./', path)


@app.route('/segment_and_predict', methods=['GET', 'POST'])
def segment_and_predict():
    # Preprocess cropped license plate image
    if request.method == 'POST':
        uploaded_file = request.files['image']
        original_image = uploaded_file.filename
        print('original_image', original_image)

        if not uploaded_file:
            return render_template('index.html', label="No file")

        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_ext = os.path.splitext(filename)[1]
            # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            #     abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            image = cv2.imread("static/images/"+uploaded_file.filename, cv2.IMREAD_COLOR)

    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]

    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
            dic[i] = c

    output = []
    for i, ch in enumerate(char_list):  # iterating over the characters
            img_ = cv2.resize(ch, (28, 28))
            img = fix_dimension(img_)
            img = img.reshape(1, 28, 28, 3)
            y_ = model.predict_classes(img)[0]  # predicting the class
            character = dic[y_]  #
            output.append(character)
    plate_number = ''.join(output)
    return render_template('results.html', prediction=plate_number)


if __name__ == '__main__':
    app.run()