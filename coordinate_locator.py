import cv2

img = cv2.imread("screen.png")  # replace with your screenshot/image file
clone = img.copy()

ROI_WIDTH = 100
ROI_HEIGHT = 100


def coord_finder(left_click, x, y, flags, param):
    global img

    if left_click == cv2.EVENT_LBUTTONDOWN:
        top_left_x = int(x - ROI_WIDTH / 2)
        top_left_y = int(y - ROI_HEIGHT / 2)

        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)

        print("ROI:")
        print("top_left_x =", top_left_x)
        print("top_left_y =", top_left_y)
        print("width =", ROI_WIDTH)
        print("height =", ROI_HEIGHT)

        img = clone.copy()

        cv2.rectangle(
            img,
            (top_left_x, top_left_y),
            (top_left_x + ROI_WIDTH, top_left_y + ROI_HEIGHT),
            (0, 255, 0),
            2
        )


cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", coord_finder)

while True:
    cv2.imshow("Select ROI", img)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()