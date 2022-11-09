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
@app.route('/', methods =["GET", "POST"])
def test_page():
    example_embed = 'Sending data... [this is text from python]'
    # look inside `templates` and serve `index.html`
    return render_template('index.html', embed=example_embed)


######## result of coordinates ############
@app.route('/coords', methods =["GET", "POST"])
def coords_page():
    if request.method == "POST":
       x_coord = request.form.get("x_coord", type = int)
       y_coord = request.form.get("y_coord")
    else:
        return

    img = "group.png"
    # x_coord = x_coord.astype(np.float32)

    if x_coord < 360:
        mask = "mask1.png"
    elif x_coord < 515:
        mask = "mask2.png"
    elif x_coord < 680:
        mask = "mask3.png"
    elif x_coord < 860:
        mask = "mask4.png"
    else:
        mask = "mask5.png"

    return alpha_blending(img, mask) 
    # return "(x, y) = (" + x_coord + ", " + y_coord + ")" 


######## work in progress: ignore this for now ############
@app.route('/rectangle')
def draw_rectangle():
    example_embed = 'Sending data... [this is text from python]'
    # look inside `templates` and serve `index.html`
    return render_template('rectangle.html', embed=example_embed)


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


@app.route('/alpha_blending/<image_path>/<mask>/<level>')
def alpha_blending(image_path, mask, level):
    if mask == "original":
        return render_template("blending.html", sample_image='src/%s' % image_path)

    original = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/%s' % image_path)
    blurred = cv.GaussianBlur(original, (int(level), int(level)), 0)

    alpha = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/mask/%s' % mask)

    blurred = cv.bitwise_and(blurred, alpha)
    background = cv.bitwise_and(original, cv.bitwise_not(alpha))
    outcome = cv.add(background, blurred)
    # outcome = cv.addWeighted(outcome, 0.8, original, 0.2, 0.0)

    # blending needed

    num = time.localtime(time.time()).tm_sec
    filename = 'alpha_%s.jpg' % str(num)
    cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, outcome)

    # blending
    size = 768
    filename = pyramid_blending(original[:size, :size], original[:size, :size])

    return render_template("blending.html", sample_image='blending/%s' % filename)


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
    return filename


def convolution(img, kernel):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    kernel_size = kernel.shape[0]
    pad_amount = int(kernel_size / 2)
    gaussian_convolved_img = np.zeros(img.shape)
    for i in range(3):
        zero_padded = img #cv.pad(img[:, :, i], u_pad=pad_amount, v_pad=pad_amount)
        for r in range(pad_amount, MAX_ROWS + pad_amount):
            for c in range(pad_amount, MAX_COLS + pad_amount):
                #             print("r-pad_amount", r-pad_amount)
                #             print("r-pad_amount+kernel_size", r-pad_amount+kernel_size)
                conv = np.multiply(zero_padded[r - pad_amount:r - pad_amount + kernel_size,
                                   c - pad_amount:c - pad_amount + kernel_size], kernel)
                conv = np.sum(conv)
                gaussian_convolved_img[r - pad_amount, c - pad_amount, i] = float(conv)
    return gaussian_convolved_img


def make_one_D_kernel(img, kernel):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    one_d_gaussian_kernel = kernel

    kernel_matrix = np.zeros((MAX_ROWS, MAX_ROWS))
    # print(kernel_matrix.shape)
    for m in range(MAX_ROWS):
        #     print(m)
        #     print(m+(len(one_d_gaussian_kernel)))
        #     print(one_d_gaussian_kernel)
        #     print()
        over = int(len(one_d_gaussian_kernel) / 2)
        mid = over
        lower = max(0, m - over)
        upper = min(m + over, MAX_ROWS)
        kernel_lower = mid - over if m - over >= 0 else abs(m - over)
        kernel_upper = mid + over if m + over < MAX_ROWS else (mid + over) - (m + over - MAX_ROWS)
        kernel_matrix[m, lower:upper] = one_d_gaussian_kernel[kernel_lower:kernel_upper]
    return kernel_matrix


def down_sample(img, factor=2):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    small_img = np.zeros((int(MAX_ROWS / 2), int(MAX_COLS / 2), 3))

    small_img[:, :, 0] = cv.resize(img[:, :, 0], [int(MAX_ROWS / 2), int(MAX_COLS / 2)])
    small_img[:, :, 1] = cv.resize(img[:, :, 1], [int(MAX_ROWS / 2), int(MAX_COLS / 2)])
    small_img[:, :, 2] = cv.resize(img[:, :, 2], [int(MAX_ROWS / 2), int(MAX_COLS / 2)])
    return small_img


def up_sample(img, factor=2):
    MAX_ROWS = img.shape[0]
    MAX_COLS = img.shape[1]
    small_img = np.zeros((int(MAX_ROWS * 2), int(MAX_COLS * 2), 3))

    small_img[:, :, 0] = cv.resize(img[:, :, 0], [int(MAX_ROWS * 2), int(MAX_COLS * 2)])
    small_img[:, :, 1] = cv.resize(img[:, :, 1], [int(MAX_ROWS * 2), int(MAX_COLS * 2)])
    small_img[:, :, 2] = cv.resize(img[:, :, 2], [int(MAX_ROWS * 2), int(MAX_COLS * 2)])
    return small_img


def one_level_laplacian(img, G):
    # generate Gaussian pyramid for Apple
    A = img.copy()

    # Gaussian blur on Apple
    blurred_A = convolution(A, G)

    # Downsample blurred A
    small_A = down_sample(blurred_A)

    # Upsample small, blurred A
    # insert zeros between pixels, then apply a gaussian low pass filter
    large_A = up_sample(small_A)
    upsampled_A = convolution(large_A, G)

    # generate Laplacian level for A
    laplace_A = A - upsampled_A

    # reconstruct A
    #     reconstruct_A = laplace_A + upsampled_A

    return small_A, upsampled_A, laplace_A


def F_transform(small_A, G):
    large_A = up_sample(small_A)
    upsampled_A = convolution(large_A, G)
    return upsampled_A


def gamma_decode(img):
    new_img = np.zeros((img.shape))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            new_img[r, c, 0] = np.power(img[r, c, 0], 1 / 1.2)
            new_img[r, c, 1] = np.power(img[r, c, 1], 1 / 1.2)
            new_img[r, c, 2] = np.power(img[r, c, 2], 1 / 1.2)
    return new_img


def pyramid():
    gaussian_kernel = np.load('gaussian-kernel.npy')


# run app
app.run(debug=True)
