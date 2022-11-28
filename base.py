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

    return blur(img, mask)
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


# @app.route('/blending/<blur_type>/<blur_target>')
# def blending(blur_type, blur_target):
#     outcome = ''
#
#     img = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/thor.png')
#     img = img[200:584, 0:384]
#     width = img.shape[1]
#     width_cutoff = width // 2
#
#     if blur_target == 'right':
#         img_left = img[:, :width_cutoff]
#         img_right = cv.blur(img[:, width_cutoff:], (5, 5))
#     else:
#         img_left = cv.blur(img[:, :width_cutoff], (5, 5))
#         img_right = img[:, width_cutoff:]
#
#     real = np.hstack((img_left, img_right))
#
#     num = time.localtime(time.time()).tm_sec
#     outcome = 'blending/real_%s.jpg' % str(num)
#     cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/%s' % outcome, real)
#
#     # blur type 0 - original
#     # blur type 1 - pyramid
#     # blur type 2 - alpha - TBC
#
#     if blur_type == '1':
#         outcome = pyramid_blending(img, real)
#     elif blur_type == '2':
#         outcome = blur(img, real)
#
#     print(outcome)
#
#     return render_template("blending.html", sample_image=outcome)


@app.route('/blur/<blend_type>/<image_path>/<mask_path>/<level>')
def blur(blend_type, image_path, mask_path, level):
    if mask_path == "original":
        return render_template("blending.html", sample_image='src/%s' % image_path)

    original = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/%s' % image_path)
    blurred = cv.GaussianBlur(original, (int(level), int(level)), 0)

    mask = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/mask/%s' % mask_path)

    if blend_type == "alpha":
        blurred = cv.bitwise_and(blurred, mask)
        background = cv.bitwise_and(original, cv.bitwise_not(mask))
        outcome = cv.add(background, blurred)
        num = time.localtime(time.time()).tm_sec
        filename = 'binary_%s.jpg' % str(num)
        cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, outcome)

    else:
        filename = pyramid(original, blurred, mask)

    return render_template("blending.html", sample_image='blending/%s' % filename)


def pyramid(img_a, img_b, m):
    size = 768
    img1 = img_a[:size, :size].copy()
    img2 = img_b[:size, :size].copy()

    # Create the mask

    mask = m[:size, :size].copy() / 255

    num_levels = 5

    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()
    # Blend the images
    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    # Reconstruct the images
    final = reconstruct(add_laplace)
    # Save the final image to the disk
    # cv2.imwrite('D:/downloads/pp2.jpg', final[num_levels])

    num = time.localtime(time.time()).tm_sec
    filename = 'blend_%s.jpg' % str(num)
    cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, final[num_levels])

    return filename


def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        # np.divide(mask, 255, out=mask)
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv.add(np.array(laplacian_pyr[i + 1], dtype='f'), laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


# run app
app.run(debug=True)
