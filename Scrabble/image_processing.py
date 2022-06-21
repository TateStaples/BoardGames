import cv2
from math import cos, sin, sqrt
import numpy as np
import pytesseract
import time


class Image_processor:
    cols = 15
    rows = 15

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.board = []
        self.formatting_function = self.warp_image
        # self.nn = NNLetterIdentifier()
        # self.imgs = None

    def run(self):
        # self.formatting_function()
        vals = self.grid()
        board = [[vals[row*self.cols + col] for col in range(self.cols)] for row in range(self.rows)]
        for row in board:
            print(row)
        return board

    '''
    def template_stuff(self, imgs):
        return [self.template_identify(img) for img in imgs]

    def template_process(self):
        # multi-scale = https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
        # images from: https://www.papertraildesign.com/free-printable-scrabble-letter-tiles-sign/
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        im_paths = [f"data/Tile_images/Scrabble-tile-{letter}-wood.jpg" for letter in alphabet]
        locations = []
        #im_paths = ["data/a_template.jpg"]
        w = h = 1
        for path, letter in zip(im_paths, alphabet):
            img_rgb = self.img
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            template = cv2.imread(path, 0)
            scale = img_gray.shape[1] / 25
            template = imutils.resize(template, width=int(scale))
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                locations.append((pt, letter))
        min_y = min(locations, key=lambda x: x[0][1])[0][1]
        min_x = min(locations, key=lambda x: x[0][0])[0][0]
        placed = [(
            (int(round((y-min_y)/h))%self.rows, int(round((x-min_x)/w))%self.cols), letter)
            for (x, y), letter in locations]
        placed = list(set(placed))
        cv2.imwrite('res.png', img_rgb)
        return placed

    def template_identify(self, img):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWYZ"  # todo add X
        #im_paths = [f"data/Tile_images/Scrabble-tile-{letter}-wood.jpg" for letter in alphabet]
        im_paths = [f"data/real_templates/{letter}.jpg" for letter in alphabet]
        blank_paths = ["triple_letter.jpg", "triple_word.jpg", "double_word.jpg", "double_letter.jpg", "blank.jpg"]
        blank_paths = ["data/real_templates/" + path for path in blank_paths]
        blanks = "     " # one for each blank path
        im_paths.extend(blank_paths)
        alphabet += blanks
        max_val = 0
        best_letter = ""
        h, w = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
        for path, letter in zip(im_paths, alphabet):
            template = cv2.imread(path, 0)
            _, template = cv2.threshold(template, 40, 255, cv2.THRESH_BINARY)
            template = imutils.resize(template, width=int(min((w, h)) * .75))
            cv2.imwrite("template.jpg", template)
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            # maxVal = self.multi_scale_template(img, template)
            #print(letter, maxVal)
            if maxVal > max_val:
                max_val = maxVal
                best_letter = letter
        print(best_letter if best_letter != " " else "blank")
        return best_letter

    def online_image(self):
        # https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 90, 150, apertureSize=3)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5,5),np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        cv2.imwrite('canny.jpg', edges)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=90, threshold=100, minLineLength=0, maxLineGap=20)
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[
                i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now
        filtered_lines = []
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
        for line in filtered_lines:
            rho, theta = line[0]
            a = cos(theta)
            b = sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite('hough.jpg', self.img)

    Previous types:
    MSER
    Canny edge detection
    brisk - https://www.youtube.com/watch?v=uwN0JAY548M
    '''
        
    def multi_scale_template(self, img, template):
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (mv, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 5)
        print(startX, endX)
        cv2.imwrite("Image.jpg", img)
        return mv

    def generate_corners(self, row, col):
        tl = np.array([[1, 1],
                        [1, 0]])
        br = np.array([[0, 1],
                       [1, 1]])
        tr = np.array([[1, 1],
                       [0, 1]])
        bl = np.array([[1, 0],
                       [1, 1]])
        tl = np.kron(tl, np.ones((row, col)))
        br = np.kron(br, np.ones((row, col)))
        tr = np.kron(tr, np.ones((row, col)))
        bl = np.kron(bl, np.ones((row, col)))
        #print(tl.shape)
        tl *= 255
        br *= 255
        bl *= 255
        tr *= 255
        tl = cv2.cvtColor(tl.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        br = cv2.cvtColor(br.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bl = cv2.cvtColor(bl.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tr = cv2.cvtColor(tr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return tl, tr, bl, br

    def warp_image(self):
        corners = self.get_corners()
        self.img = self.perspective_warp(corners, self.img)
        cv2.imwrite("result.jpg", self.img)

    def grid(self):
        self.imgs = self.crop(self.img, self.rows, self.cols)
        return self.identify(self.imgs)
        # self.draw_grid(img, self.rows, self.cols)

    def get_corners(self):
        size = 27
        tl, tr, bl, br = self.generate_corners(size, size)
        corners = tl, tr, br, bl
        size *= 2
        points = []
        for i, corner in enumerate(corners):
            dx = size if i==0 or i==3 else 0
            dy = size if i==0 or i==1 else 0
            result = cv2.matchTemplate(self.img, corner, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            x, y = maxLoc
            rect(self.img, x, y, size, size, (0, 255, 0))
            points.append((x+dx, y+dy))
        cv2.imwrite("corners.jpg", self.img)
        return points

    @staticmethod
    def draw_grid(img, rows, cols):
        h, w = img.shape[:2]
        x_step = w/cols
        y_step = h/rows
        for c in range(1, cols):
            x = int(c * x_step)
            cv2.line(img, (x, 0), (x, h-1), (0, 255, 0), 3)
        for r in range(1, rows):
            y = int(r*y_step)
            cv2.line(img, (0, y), (w-1, y), (0, 255, 0), 3)
        cv2.imwrite("grid.jpg", img)

    @staticmethod
    def crop(img, rows, cols):
        h, w = img.shape[:2]
        x_step = w / cols
        y_step = h / rows
        imgs = []
        for r in range(rows):
            for c in range(cols):
                x1 = int(c * x_step)
                y1 = int(r * y_step)
                x2 = int(x1 + x_step)
                y2 = int(y1 + y_step)
                section = img[y1: y2, x1: x2]
                imgs.append(section)
        return imgs

    @staticmethod
    def padded_crop(img, rows, cols, pad=.5):
        h, w = img.shape[:2]
        x_step = w / cols
        y_step = h / rows
        imgs = []
        for r in range(rows):
            for c in range(cols):
                x1 = int(c * x_step) if c == 0 else int((c-pad) * x_step)
                y1 = int(r * y_step) if r == 0 else int((r-pad) * y_step)
                x2 = int(x1 + x_step) if c == cols-1 else int((c+1+pad) * x_step)
                y2 = int(y1 + y_step) if r == rows-1 else int((r+1+pad) * y_step)
                section = img[y1: y2, x1: x2]
                imgs.append(section)
        return imgs

    def perspective_warp(self, corners, img):
        pts = np.array(corners, dtype="float32")
        tl, tr, br, bl = corners
        widthA = np.sqrt((br[0] - bl[0])**2 + (br[1]-bl[1]) ** 2)
        widthB = np.sqrt((tr[1] - tl[1])**2 + (tr[0]-tl[0]) ** 2)
        width = int(max(widthA, widthB))
        heightA = np.sqrt((br[0] - tr[0])**2 + (br[1]-tr[1]) ** 2)
        heightB = np.sqrt((bl[1] - tl[1])**2 + (bl[0]-tl[0]) ** 2)
        height = int(max(heightA, heightB))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")
        transform = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, transform, (width, height))
        return warped

    def get_sub_img(self, row, col):
        return self.imgs[row * 15 + col]

    def identify(self, imgs):
        check_color = True
        check_dark = True
        config = '--psm 8'
        colors = {(60, 65, 150): '3w', (90, 80, 60): '3l', (135, 140, 175): '2w', (125, 125, 115): '2l'}

        results = []
        t = time.time()
        for i, img in enumerate(imgs):
            formatted = self.format(img)
            if check_dark:
                w, h = formatted.shape
                dw, dh = w//5, h//5
                shrunk = 255 - formatted[dw:w-dw, dh:h-dh]
                dark = shrunk.sum()//255  # number of dark pixels
                if dark < 200:
                    results.append('')
                    continue
            if check_color:
                buffer = 10
                found = False
                # get the dominant colors
                _, labels, palette = cv2.kmeans(np.float32(img.reshape(-1, 3)), 1, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1), 10, cv2.KMEANS_RANDOM_CENTERS)
                _, counts = np.unique(labels, return_counts=True)
                dominant = palette[np.argmax(counts)]
                for color in colors:  # get the euclidean distance
                    if found: break
                    total = 0
                    for c1, c2 in zip(color, dominant):
                        total += (c1-c2) ** 2
                    if sqrt(total) < buffer:
                        results.append(colors[color])
                        break
                # results.append(tuple(dominant))
                if found: continue
            result = pytesseract.image_to_string(formatted, config=config).strip()
            results.append(result)
        print("Identify time:", time.time()-t)
        return results
        # return self.nn.identify(imgs)

    def format(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, formatted = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # formatted = cv2.erode(formatted, kernel=(3, 3), iterations=1)
        return formatted


def rect(i, x, y, w, h, c):
    cv2.rectangle(i, (x, y), (x+w, y+h), c)


class NNLetterIdentifier:
    train = False
    save_path = "data/nn_model"
    base_decode = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    modified_decode = "oi****g***abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"

    def __init__(self):
        self.nn = None
        # if self.train:
        #     self.create_neural_network()
        # else:
        #     self.nn = keras.models.load_model(self.save_path)

    def create_neural_network(self):  # could replace with Hu moment recognition
        data, labels = self.load_data()
        model = keras.Sequential([  # replace with CNN
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(500, activation="relu"),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(62, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(data, labels, epochs=4)
        self.nn = model
        model.save(self.save_path)

    @staticmethod
    def load_data():
        # https://github.com/akshaybahadur21/Alphabet-Recognition-EMNIST/blob/master/Alpha-Rec.py
        emnist_data = MNIST(path='data/', return_type='numpy')
        imgs, labels = emnist.extract_training_samples('byclass')
        imgs, labels = emnist_data.load_training()
        # return imgs, labels

    def load_bad_data(self):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWYZ"
        print(alphabet[24])
        im_paths = [f"data/real_templates/{letter}.jpg" for letter in alphabet]
        imgs = [cv2.imread(im_path) for im_path in im_paths]
        data = np.array([self.process_images(imgs)])
        labels = np.array(list(range(1, 27)))
        return data, labels

    @staticmethod
    def process_images(imgs):
        images = []
        kernel = np.ones((3, 3), np.uint8)
        cutoff = 0.2
        for img in imgs:
            img = cv2.resize(img, (150, 150))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.dilate(gray, kernel, iterations=2)
            gray = cv2.erode(gray, kernel, iterations=3)
            _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            width, height = bw.shape
            v1 = int(min(width, height) * cutoff)
            bw = bw[v1: height-v1, v1:width-v1]
            shrunk = cv2.resize(bw, (28, 28))
            shrunk = 255-shrunk
            # shrunk = np.reshape(shrunk, 784)
            images.append(shrunk)
        return np.asarray(images)

    @staticmethod
    def decode(output):
        index = np.argmax(output)
        letter = NNLetterIdentifier.modified_decode[index] + '/' + NNLetterIdentifier.base_decode[index]
        return letter

    def identify(self, imgs):
        inputs = self.process_images(imgs)
        return [self.predict(img) for img in inputs]

    def predict(self, img):
        threshold = 57
        count = np.count_nonzero(img)
        if count < threshold:  # checks if blank
            return "_/_"
        return self.decode(self.nn.predict(np.array([img]))[0])

if __name__ == '__main__':
    img_path = "result.jpg"
    i = Image_processor(img_path)
    i.run()
    # Image_processor.draw_grid()
    # cv2.imwrite("section.jpg", i.get_sub_img(4, 3))
    # cv2.imwrite("nn.jpg", NNLetterIdentifier.process_images([i.get_sub_img(7, 9)])[0])
    # img_path = 'section.jpg'
    # img = cv2.imread(img_path)
    # formatted = Image_processor.format(None, img)
    # print((255-formatted).sum()//255)
    # cv2.imwrite('Image.jpg', formatted)
    # letter = pytesseract.image_to_string(formatted, config='--psm 10').strip()
    # print(letter)
    # nn = NNLetterIdentifier()
    # print(nn.identify([img]))
    # processed = NNLetterIdentifier.process_images([img])[0]
    # cv2.imwrite("nn.jpg", processed)

