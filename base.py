# app.py
from flask import Flask, jsonify, request, render_template
import time
import cv2 as cv
import numpy as np

app = Flask(__name__)

global graph
global cycles
global color
global par
global cyclenumber

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


@app.route('/companion/<target>')
def companion(target):
    boxes = np.array([[42.319610595703125, 118.13470458984375, 702.53466796875, 369.06103515625],
                      [204.07962036132812, 844.4298095703125, 619.9171142578125, 1112.060791015625],
                      [131.9391632080078, 513.0297241210938, 611.9575805664062, 678.2179565429688],
                      [254.4031982421875, 382.82110595703125, 695.7974853515625, 574.1754760742188],
                      [285.6714782714844, 613.717041015625, 698.4228515625, 842.5339965820312]])
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    classes = np.array([1, 1, 1, 1, 1])

    people = filter_boxes(boxes, classes, scores)

    strangers = set(range(len(people))) - companionDetection(people, target, 0) # people not in companions

    image_path = 'group.png'
    level = 9
    original = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/src/%s' % image_path)
    blurred = cv.GaussianBlur(original, (int(level), int(level)), 0)
    mask = np.zeros(original.shape, dtype=np.uint8)

    for s in strangers:
        n = cv.imread('/Users/owner/Documents/GitHub/companion-detector/static/mask/s%s.jpg' % s)
        mask = cv.bitwise_or(mask, n)

    filename = pyramid_blending(original, blurred, mask)

    return render_template("blending.html", sample_image='blending/%s' % filename)


def filter_boxes(boxes, classes, scores):
    people = []
    for b, c, s in zip(boxes, classes, scores):
        if c == 1 and s > 0.8:
            people.append(b)

    return people


def pyramid_blending(img_a, img_b, m):
    img_a[:img_a.shape[0]//2, :] = pyramid(img_a, img_b, m, 100)[:img_a.shape[0]//2, :]
    img_a[:img_a.shape[0]//2, :] = pyramid(img_a, img_b, m, 800)[:img_a.shape[0]//2, :]

    num = time.localtime(time.time()).tm_sec
    filename = 'blend_%s.jpg' % str(num)
    print(filename)
    cv.imwrite('/Users/owner/Documents/GitHub/companion-detector/static/blending/%s' % filename, img_a)

    return filename


def pyramid(img_a, img_b, m, x = 800):
    # x,y coordinates from either button click or box
    size = 256*3
    if x < img_a.shape[0]/2:
        x0, x1, y0, y1 = 0, size, 0, size
    else:
        x0, x1, y0, y1 = img_a.shape[0]-size, img_a.shape[0], img_a.shape[1]-size, img_a.shape[1]
    img1 = img_a[x0:x1, y0:y1].copy()
    img2 = img_b[x0:x1, y0:y1].copy()
    mask = m[x0:x1, y0:y1].copy() / 255

    num_levels = 7

    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)

    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

    mask_pyr_final = gaussian_pyramid(cv.GaussianBlur(mask, (9, 9), 9), num_levels)
    mask_pyr_final.reverse()

    # Blend the images
    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    final = reconstruct(add_laplace)

    img_a[x0:x1, y0:y1] = final[num_levels]

    return img_a


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


def companionDetection(ps, target, offset):
    global graph
    global cycles
    global color
    global par
    global cyclenumber

    graph = [[] for i in range(len(ps))]
    cycles = [[] for i in range(len(ps))]
    color = [0] * len(ps)
    par = [0] * len(ps)

    # store the numbers of cycle
    cyclenumber = 0
    companions = graphing(ps, target, offset)
    print(companions)
    return companions
    # print_boxes_with_index(img_path, ps, companions)


def graphing(boxes, target_idx, proximity_max):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if (isAdjacent(boxes[i], boxes[j], proximity_max)):
                addEdge(graph, i, j)
    # store the numbers of cycle

    # call DFS to mark the cycles
    dfs_cycle(1, 0, color, par)

    return collectNodes(target_idx)


def collectNodes(target):
    nodes = set()
    nodes.add(int(target))

    ns = nodes.copy()
    for n in ns:
        for e in graph[n]:
            nodes.add(e)

    for i in range(0, cyclenumber):

        if target not in cycles[i]:
            continue
        for x in cycles[i]:
            nodes.add(x)

    ns = nodes.copy()

    for n in ns:
        for e in graph[n]:
            nodes.add(e)
    return nodes


def dfs_cycle(u, p, color: list, par: list):
    global cyclenumber

    if color[u] == 2:
        return

    if color[u] == 1:
        v = []
        cur = p
        v.append(cur)

        while cur != u:
            cur = par[cur]
            v.append(cur)
        cycles[cyclenumber] = v
        cyclenumber += 1

        return

    par[u] = p

    color[u] = 1

    for v in graph[u]:

        if v == par[u]:
            continue
        dfs_cycle(v, u, color, par)

    color[u] = 2


def addEdge(g, u, v):
    g[u].append(v)
    g[v].append(u)


def isAdjacent(boxA, boxB, proximity_max):
    return (boxA[0] < boxB[2] + proximity_max and boxA[1]< boxB[3] + proximity_max and boxA[2] + proximity_max > boxB[0] and boxA[3] + proximity_max > boxB[1]) or (boxA[0] < boxB[2] + proximity_max and boxA[1]< boxB[3] + proximity_max and boxA[2] + proximity_max > boxB[0] and boxA[3] + proximity_max > boxB[1])


# run app
app.run(debug=True)
