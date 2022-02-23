import numpy as np
from keras.preprocessing import image
import cv2
from matplotlib.pyplot import imread
import sys
import os
import flask
from flask import Flask, render_template, request, flash, send_from_directory
from keras.models import load_model
from keras.models import model_from_json
import secrets
import glob
import matplotlib.pyplot as plt


secret = secrets.token_urlsafe(32)

sys.path.append(os.path.abspath("./model2"))

app = Flask(__name__)
app.secret_key = secret

app.config['UPLOAD_FOLDER'] = './'

#dict1 = {0: 'Musang_King', 1: 'Golden_pheonix', 2: '101', 3: 'Kucing_tidur', 4: 'Black_thorn'}

dict1 = {0: '101', 1: 'D13', 2: 'KH', 3: 'D24', 4: 'D88', 5: 'D2', 6: 'Black_thorn', 7: 'Golden_pheonix', 8: 'Musang_king', 9: 'Kucing_tidur'}

model1 = model_from_json(open('model2.json').read())
model1.load_weights('model2.h5')
model1.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])



# routes
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='bg.jpg')
                               
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "nice"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        if 'file1' not in request.files:
            flash('No file part')
            print('Error')
            return 'Error'
        x = request.files['file1']
        #print(type(x), x)
        path = os.path.join(app.config['UPLOAD_FOLDER'], x.filename)
        x.save(path)

        import matplotlib.cbook as cbook
        import matplotlib.image as image
        import matplotlib.pyplot as plt

        data = []
        img = plt.imread(x.filename)
        print(x.filename)
        data.append(img)

        image = data[0]

        x = image
        x = np.invert(x)
        x = cv2.resize(x, (200, 200))
        x = x.reshape(1, 200, 200, 3)
        out = model1.predict(x)
        response = np.array_str(np.argmax(out, axis=1))
        # print(type(response[1]))
        return render_template('result.html', prediction=int(response[1]))


if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(debug = True)
    
