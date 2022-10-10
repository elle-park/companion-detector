# app.py
from flask import Flask, jsonify, request, render_template
import time
import cv2 as cv
import numpy as np

app = Flask(__name__)

######## Example data, in sets of 3 ############
data = list(range(1, 300, 3))
print(data)


######## HOME ############
@app.route('/')
def test_page():
    example_embed = 'Sending data... [this is text from python]'
    # look inside `templates` and serve `index.html`
    return render_template('index.html', embed=example_embed)


######## Example fetch ############
@app.route('/test', methods=['GET', 'POST'])
def testfn():
    # POST request
    if request.method == 'POST':
        print(request.get_json())  # parse as JSON
        return 'OK', 200
    # GET request
    else:
        message = {'greeting': 'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers


######## Data fetch ############
@app.route('/getdata/<transaction_id>/<second_arg>', methods=['GET', 'POST'])
def datafind(transaction_id, second_arg):
    # POST request
    if request.method == 'POST':
        print('Incoming..')
        print(request.get_text())  # parse as text
        return 'OK', 200
    # GET request
    else:
        message = 't_in = %s ; result: %s ; opt_arg: %s' % (transaction_id, data[int(transaction_id)], second_arg)
        return message  # jsonify(message)  # serialize and use JSON headers


@app.route('/blending/<blur_type>/<blur_target>')
def blending(blur_type, blur_target):
    outcome = ''

    img = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/thor.png')
    img = img[200:584, 0:384]
    width = img.shape[1]
    width_cutoff = width // 2

    if blur_target == 'right':
        img_left = img[:, :width_cutoff]
        img_right = cv.blur(img[:, width_cutoff:], (5, 5))
    else:
        img_left = cv.blur(img[:, :width_cutoff], (5, 5))
        img_right = img[:, width_cutoff:]

    real = np.hstack((img_left, img_right))

    num = time.localtime(time.time()).tm_sec
    outcome = 'blending/real_%s.jpg' % str(num)
    cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/%s' % outcome, real)

    # blur type 0 - original
    # blur type 1 - pyramid
    # blur type 2 - alpha - TBC
    if blur_type == '1':
        outcome = pyramid_blending(img, real)
    elif blur_type == '2':
        outcome = alpha_blending(img, real)

    print(outcome)

    return render_template("blending.html", sample_image=outcome)


def alpha_blending(A, B):
    return "outcome.jpg"


def pyramid_blending(A, B):
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)

    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gpA[i])
        L = cv.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i - 1], GE)
        lpB.append(L)

    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])

    num = time.localtime(time.time()).tm_sec
    filename = 'pyramid_%s.jpg' % str(num)
    cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, ls_)
    return 'blending/%s' % filename


# run app
app.run(debug=True)
