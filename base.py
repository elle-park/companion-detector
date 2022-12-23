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
    return render_template('sample1.html', embed=example_embed)

@app.route('/sample2', methods =["GET", "POST"])
def sample2_page():
    example_embed = 'Sending data... [this is text from python]'
    # look inside `templates` and serve `index.html`
    return render_template('sample2.html', embed=example_embed)

@app.route('/sample3', methods =["GET", "POST"])
def sample3_page():
    example_embed = 'Sending data... [this is text from python]'
    return render_template('sample3.html', embed=example_embed)

@app.route('/sample4', methods =["GET", "POST"])
def sample4_page():
    example_embed = 'Sending data... [this is text from python]'
    return render_template('sample4.html', embed=example_embed)

def select_mask_from_coords(x_point, y_point, mask_coords):
    
    filename = 'ERROR'
    coord = []
    for file, m in mask_coords.items():
      [y_left_top, x_left_top, y_right_bttm, x_right_bttm] = m[0:4]

      if ((y_left_top <= y_point <= y_right_bttm) and ((x_left_top <= x_point <= x_right_bttm))):
        filename = file
        coord = [y_left_top, x_left_top, y_right_bttm, x_right_bttm]
    
    target = list(mask_coords).index(filename)  
      # return filename + index in mask_coords of that filename
    return filename, coord, target

######## result of coordinates ############
@app.route('/slider', methods =["GET", "POST"])
def slider_page():
    
    if request.method == "POST":
        sample_directory = request.form.get("sample_page", type = str)
        x_coord = request.form.get("x_coord", type = int)
        y_coord = request.form.get("y_coord", type = int)
        blurlvl = request.form.get("blur_level", type = int)
        blendlvl = request.form.get("gradient_level", type = int)
        proximitylvl = request.form.get("proximity_level", type = int)
        custom_levels = [blurlvl, blendlvl, proximitylvl]
       
    else:
        return

    img, s_coords = get_image_info(sample_directory)
    
    mask, mask_coords, target = select_mask_from_coords(x_coord, y_coord, s_coords)

    return companion(target, custom_levels, img, sample_directory, s_coords)

def get_image_info(sample):
    img = ""
    coordinfo = {}
    
    if (sample == "sample-1"):
        img = "sample-1.png"
        s0_coords = [42.319610595703125, 118.13470458984375, 702.53466796875, 369.06103515625]
        s1_coords = [204.07962036132812, 844.4298095703125, 619.9171142578125, 1112.060791015625]
        s2_coords = [131.9391632080078, 513.0297241210938, 611.9575805664062, 678.2179565429688]
        s3_coords = [254.4031982421875, 382.82110595703125, 695.7974853515625, 574.1754760742188]
        s5_coords = [285.6714782714844, 613.717041015625, 698.4228515625, 842.5339965820312]
        coordinfo = {'s0.png': s0_coords, 's1.png': s1_coords, 's2.png': s2_coords, 's3.png': s3_coords, 's4.png': s5_coords}
    elif (sample == "sample-2"):
        img = "sample-2.png"
        s0_coords = [243.67710876464844, 224.81951904296875, 1070.5166015625, 511.734375]
        s1_coords = [217.54873657226562, 1231.3968505859375, 1071.3770751953125, 1482.9814453125]
        s2_coords = [424.736328125, 879.4990844726562, 1078.1248779296875, 1367.6490478515625]
        s3_coords = [221.42893981933594, 1442.2025146484375, 1074.092529296875, 1683.7462158203125]
        s4_coords = [394.02789306640625, 499.1584167480469, 1028.00341796875, 817.9619140625]
        s5_coords = [276.03204345703125, 725.0794067382812, 799.4697875976562, 952.715576171875]
        coordinfo = {'s0.png': s0_coords, 's1.png': s1_coords, 's2.png': s2_coords, 's3.png': s3_coords, 's4.png': s4_coords, 's5.png': s5_coords}
    elif (sample == "sample-3"):
        img = "sample-3.jpg"
        s0_coords = [256.8323059082031, 583.7996826171875, 504.3184814453125, 674.3363647460938]
        s1_coords = [108.08509826660156, 200.8914794921875, 862.708251953125, 426.71429443359375]
        s2_coords = [160.4375762939453, 380.9007263183594, 876.4453125, 569.8856811523438]
        coordinfo = {'s0.png': s0_coords, 's1.png': s1_coords, 's2.png': s2_coords}
    elif (sample == "sample-4"):
        img = "sample-4.png"
        s0_coords = [770.14404296875, 689.4024658203125, 1237.430908203125, 888.3008422851562]
        s1_coords = [750.8616943359375, 337.7422180175781, 1260.5721435546875, 503.223388671875]
        s2_coords = [812.82958984375, 137.4326629638672, 1356.9639892578125, 322.8211669921875]
        s3_coords = [760.7828979492188, 513.8840942382812, 1163.213623046875, 652.2471923828125]
        s4_coords = [841.9423828125, 1.292824625968933, 1442.6553955078125, 118.81682586669922]
        s5_coords = [722.2968139648438, 237.93463134765625, 1162.828369140625, 364.7845764160156]
        coordinfo = {'s0.png': s0_coords, 's1.png': s1_coords, 's2.png': s2_coords, 's3.png': s3_coords, 's4.png': s4_coords, 's5.png': s5_coords}
    
    return img, coordinfo
        

        
@app.route('/blur/<blend_type>/<image_path>/<mask_path>/<level>')
def blur(image_path, mask_path, sample_directory, level):
    if mask_path == "original":
        return render_template("blending.html", sample_image='src/%s' % image_path)

    original = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/src/%s' % image_path)
    blurred = cv.GaussianBlur(original, (int(level[0]), int(level[0])), 0)

    mask = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/'+ sample_directory +'/%s' % mask_path)

    filename = pyramid_blending(original, blurred, mask, level[1])

    return render_template("blending.html", sample_image='blending/%s' % filename)



def companion(target, customlvls, image_path, sample_directory, mask_coords):
    boxes = np.array(list(mask_coords.values()))
    scores = np.full(shape=len(boxes), fill_value=0.9, dtype=np.double)
    classes = np.full(shape=len(boxes), fill_value=1, dtype=int)

    people = filter_boxes(boxes, classes, scores)
    strangers = set(range(len(people))) - companionDetection(people, target, customlvls[2]) # people not in companions

    original = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/src/%s' % image_path)

    mask = np.zeros(original.shape, dtype=np.uint8)

    for s in strangers:
        n = cv.imread('/Users/ellepark/Documents/GitHub/companion-detector/static/' + sample_directory + '/s%s.png' % s)
        mask = cv.bitwise_or(mask, n)

    cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/' + sample_directory + '/%s' % 'cmask.jpeg', mask)

    maskpath = 'cmask.jpeg'
    return blur(image_path, maskpath, sample_directory, customlvls) 


def filter_boxes(boxes, classes, scores):
    people = []
    for b, c, s in zip(boxes, classes, scores):
        if c == 1 and s > 0.8:
            people.append(b)

    return people


def pyramid_blending(img_a, img_b, m, blend_level):
    img_a[:img_a.shape[0]//2, :] = pyramid(img_a, img_b, m, blend_level, 100)[:img_a.shape[0]//2, :]
    img_a[:img_a.shape[0]//2, :] = pyramid(img_a, img_b, m, blend_level, 800)[:img_a.shape[0]//2, :]

    num = time.localtime(time.time()).tm_sec
    filename = 'blend_%s.jpg' % str(num)
    print(filename)
    cv.imwrite('/Users/ellepark/Documents/GitHub/companion-detector/static/blending/%s' % filename, img_a)

    return filename


def pyramid(img_a, img_b, m, blendlvl, x = 800):
    # x,y coordinates from either button click or box
    # size = 256*3
    size = 256*(img_a.shape[0]//256)
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

    mask_pyr_final = gaussian_pyramid(cv.GaussianBlur(mask, (blendlvl, blendlvl), blendlvl), num_levels)
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
    if offset == 0:
        return set([int(target)])
    
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
