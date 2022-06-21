import cv2
import numpy as np
import digit_recognition

nn = digit_recognition.NeuralNetwork(design=[784,200,100,10],
                                     weights=list(np.load('DR_Data/my_network.npy', allow_pickle=True)),
                                     bias=True)


def process(img, previous=None):
    print("in")
    corners = get_corners(img)
    print("hello?")
    transform, warped = warp_img(img, corners)
    print("done warp")
    sections = get_sections(warped)
    print("have sections")
    predictions = [identify(sec) for sec in sections]
    same_as_previous = previous is not None and all([sec2 is None or sec1 == sec2 for sec1, sec2 in zip(predictions, previous)])
    if same_as_previous:
        # insert images
        new_insert = draw_solutions(warped, predictions)
        reverse_transform = np.linalg.inv(transform)
        warped = cv2.warpPerspective(new_insert, reverse_transform, (img.shape[1], img.shape[0]))
        cv2.fillConvexPoly(img, np.ceil(corners).astype(int), 0, 16)
        img = img + warped
        cv2.imshow(img)
        return predictions
    return predictions


def identify(section):
    resized = np.resize(section, (28, 28))
    input = np.asarray(resized)[:,:,0] * (0.99/255) + 0.01
    if input.sum() > 0.01 * 784:
        output = np.argmax(nn.run(input).T[0])
    else:
        output = None
    return output


def draw_solutions(warped, solutions):
    w, h = warped.shape
    w_step, h_step = w // 9, h // 9
    for r, row in enumerate(solutions):
        for c, solution in enumerate(row):
            x, y = (c+.5) * w_step, (h+.5) * h_step
            warped = cv2.putText(warped, str(solution), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    return warped


def get_sections(formatted):
    size = 9
    w, h = formatted.shape
    w_step, h_step = w//9, h//9
    sections = []
    for row in range(size):
        for col in range(size):
            subsection = formatted[row*h_step: (row+1)*h_step-1, col*w_step: (col+1)*w_step-1]
            sections.append(subsection)
    return sections


def warp_img(img, corners):
    pts = np.float32(corners)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    trans = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, trans, (square, square))
    return trans, warped


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners


if __name__ == '__main__':
    net = digit_recognition.run_gui()