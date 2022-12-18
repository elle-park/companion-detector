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

######## HOME ############
@app.route('/sample1', methods =["GET", "POST"])
def sample1_page():
    example_embed = 'Sending data... [this is text from python]'
    # coords = [y_left_top, x_left_top, y_right_bttm, y_right_bttm]
    s0_coords = [42.319610595703125, 118.13470458984375, 702.53466796875, 369.06103515625]
    s1_coords = [204.07962036132812, 844.4298095703125, 619.9171142578125, 1112.060791015625]
    s2_coords = [131.9391632080078, 513.0297241210938, 611.9575805664062, 678.2179565429688]
    s3_coords = [254.4031982421875, 382.82110595703125, 695.7974853515625, 574.1754760742188]
    s5_coords = [285.6714782714844, 613.717041015625, 698.4228515625, 842.5339965820312]
    mask_coords = {'s0.png': s0_coords, 's1.png': s1_coords, 's2.png': s2_coords, 's3.png': s3_coords, 's5.png': s5_coords}

    # look inside `templates` and serve `index.html`
    return render_template('sample1.html', embed=example_embed)

@app.route('/sample2', methods =["GET", "POST"])
def sample2_page():
    example_embed = 'Sending data... [this is text from python]'
    # look inside `templates` and serve `index.html`
    return render_template('sample2.html', embed=example_embed)


######## result of coordinates ############
@app.route('/slider', methods =["GET", "POST"])
def slider_page():
    blendType = 'pyr'
    
    if request.method == "POST":
       x_coord = request.form.get("x_coord", type = int)
       y_coord = request.form.get("y_coord", type = int)
       blurlvl = request.form.get("blur_level", type = int)
       
    else:
        return

    img = "group.png"
    # x_coord = x_coord.astype(np.float32)

    if x_coord < 360:
        mask = "s0.jpeg"
    elif x_coord < 515:
        mask = "s3.jpeg"
    elif x_coord < 680:
        mask = "s2.jpeg"
    elif x_coord < 860:
        mask = "s4.jpeg"
    else:
        mask = "s1.jpeg"

    return blur(blendType, img, mask, blurlvl) 


# @app.route('/alpha_blending/<image_path>/<mask>')
# def alpha_blending(image_path, mask):
#     if mask == "original":
#         return render_template("blending.html", sample_image='src/%s' % image_path)

#     img = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/src/%s' % image_path).astype(np.float32)
#     blurred = cv.blur(img, (11, 11)).astype(np.float32)

#     alpha = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/mask/%s' % mask).astype(np.float32)/255

#     blurred = cv.multiply(1-alpha, blurred)
#     img = cv.multiply(alpha, img)

#     outcome = cv.add(img, blurred)
#     num = time.localtime(time.time()).tm_sec
#     filename = 'alpha_%s.jpg' % str(num)
#     cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/blending/%s' % filename, outcome)

#     return render_template("blending.html", sample_image='blending/%s' % filename)



# @app.route('/alpha_blending/<image_path>/<mask>')
# def alpha_blending_blurred(image_path, mask, blurlvl):
#     if mask == "original":
#         return render_template("blending.html", sample_image='src/%s' % image_path)

#     original = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/src/%s' % image_path)
#     blurred = cv.GaussianBlur(original, (int(blurlvl), int(blurlvl)), 0)

#     alpha = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/mask/%s' % mask)

#     blurred = cv.bitwise_and(blurred, alpha)
#     background = cv.bitwise_and(original, cv.bitwise_not(alpha))
#     outcome = cv.add(background, blurred)
#     outcome = cv.addWeighted(outcome, 0.8, original, 0.2, 0.0)

#     # blending needed

#     num = time.localtime(time.time()).tm_sec
#     filename = 'alpha_%s.jpg' % str(num)
#     cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/blending/%s' % filename, outcome)

#     # blending
#     size = 768

#     return render_template("blending.html", sample_image='blending/%s' % filename)


# def pyramid_blending(A, B):
#     G = A.copy()
#     gpA = [G]
#     for i in range(6):
#         G = cv.pyrDown(G)
#         gpA.append(G)
#     # generate Gaussian pyramid for B
#     G = B.copy()
#     gpB = [G]
#     for i in range(6):
#         G = cv.pyrDown(G)
#         gpB.append(G)

#     lpA = [gpA[5]]
#     for i in range(5, 0, -1):
#         GE = cv.pyrUp(gpA[i])
#         L = cv.subtract(gpA[i - 1], GE)
#         lpA.append(L)

#     # generate Laplacian Pyramid for B
#     lpB = [gpB[5]]
#     for i in range(5, 0, -1):
#         GE = cv.pyrUp(gpB[i])
#         L = cv.subtract(gpB[i - 1], GE)
#         lpB.append(L)

#     LS = []
#     for la, lb in zip(lpA, lpB):
#         rows, cols, dpt = la.shape
#         ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
#         LS.append(ls)

#     ls_ = LS[0]
#     for i in range(1, 6):
#         ls_ = cv.pyrUp(ls_)
#         ls_ = cv.add(ls_, LS[i])

#     num = time.localtime(time.time()).tm_sec
#     filename = 'pyramid_%s.jpg' % str(num)
#     cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, ls_)
#     return 'blending/%s' % filename

######## result of coordinates ############
# @app.route('/coords', methods =["GET", "POST"])
# def coords_page():
#     if request.method == "POST":
#        x_coord = request.form.get("x_coord", type = int)
#        y_coord = request.form.get("y_coord")
       
#     else:
#         return

#     img = "group.png"
#     # x_coord = x_coord.astype(np.float32)

#     if x_coord < 360:
#         mask = "s0.jpeg"
#     elif x_coord < 515:
#         mask = "s3.jpeg"
#     elif x_coord < 680:
#         mask = "s2.jpeg"
#     elif x_coord < 860:
#         mask = "s4.jpeg"
#     else:
#         mask = "s1.jpeg"

#     return alpha_blending(img, mask) 
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

#     img = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/thor.png')
#     img = img[200:584, 0:384]
#     width = img.shape[1]
#     width_cutoff = width // 2

#     if blur_target == 'right':
#         img_left = img[:, :width_cutoff]
#         img_right = cv.blur(img[:, width_cutoff:], (5, 5))
#     else:
#         img_left = cv.blur(img[:, :width_cutoff], (5, 5))
#         img_right = img[:, width_cutoff:]

#     real = np.hstack((img_left, img_right))

#     num = time.localtime(time.time()).tm_sec
#     outcome = 'blending/real_%s.jpg' % str(num)
#     cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/%s' % outcome, real)

#     # blur type 0 - original
#     # blur type 1 - pyramid
#     # blur type 2 - alpha - TBC

#     if blur_type == '1':
#         outcome = pyramid_blending(img, real)
#     elif blur_type == '2':
#         outcome = alpha_blending(img, real)

#     print(outcome)

#     return render_template("blending.html", sample_image=outcome)

@app.route('/blur/<blend_type>/<image_path>/<mask_path>/<level>')
def blur(blend_type, image_path, mask_path, level):
    if mask_path == "original":
        return render_template("blending.html", sample_image='src/%s' % image_path)

    original = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/src/%s' % image_path)
    blurred = cv.GaussianBlur(original, (int(level), int(level)), 0)

    mask = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/mask/%s' % mask_path)

    if blend_type == "alpha":
        blurred = cv.bitwise_and(blurred, mask)
        background = cv.bitwise_and(original, cv.bitwise_not(mask))
        outcome = cv.add(background, blurred)
        num = time.localtime(time.time()).tm_sec
        filename = 'binary_%s.jpg' % str(num)
        cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/blending/%s' % filename, outcome)

    else:
        filename = pyramid(original, blurred, mask)

    return render_template("blending.html", sample_image='blending/%s' % filename)

def pyramid(img_a, img_b, m):
    size = 768
    img1 = img_a[:size, :size].copy()
    img2 = img_b[:size, :size].copy()

    mask = m[:size, :size].copy() / 255

    num_levels = 7

    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)

    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    # change this for blend gradient level (from 9 to some level)
    mask_pyr_final = gaussian_pyramid(cv.GaussianBlur(mask, (9,9), 9), num_levels)
    mask_pyr_final.reverse()


    # Blend the images
    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    final = reconstruct(add_laplace)

    num = time.localtime(time.time()).tm_sec
    filename = 'blend_%s.jpg' % str(num)
    print(filename)
    cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/blending/%s' % filename, final[num_levels])

    return filename

def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv.GaussianBlur(lower, (13, 13), 13)
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

def pyr(img, num_levels):
    lower = img.copy()
    smaller = img.copy()

    gaussian_pyr = [lower]
    pre_blur_pyr = [smaller]
    for i in range(num_levels):
        lower = cv.GaussianBlur(lower, (9, 9), 13)
        lower = cv.pyrDown(lower)
        smaller = cv.pyrDown(smaller)
        gaussian_pyr.append(np.float32(lower))
        pre_blur_pyr.append(np.float32(smaller))

    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(pre_blur_pyr[i - 1], gaussian_expanded)
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
