import cv2
import numpy as np

def empty(a):
    pass

class HandDetection():
    def __init__(self):
        pass

    def create_trackbars(self):
        cv2.namedWindow('Trackbars')
        cv2.resizeWindow('Trackbars', 500, 300)
        cv2.createTrackbar('HueMin', 'Trackbars', 0, 179, empty)
        cv2.createTrackbar('HueMax', 'Trackbars', 39, 179, empty)
        cv2.createTrackbar('SatMin', 'Trackbars', 51, 255, empty)
        cv2.createTrackbar('SatMax', 'Trackbars', 255, 255, empty)
        cv2.createTrackbar('ValMin', 'Trackbars', 0, 255, empty)
        cv2.createTrackbar('ValMax', 'Trackbars', 254, 255, empty)
    
    def create_mask(self, img):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_min = cv2.getTrackbarPos('HueMin', 'Trackbars')
        hue_max = cv2.getTrackbarPos('HueMax', 'Trackbars')
        sat_min = cv2.getTrackbarPos('SatMin', 'Trackbars')
        sat_max = cv2.getTrackbarPos('SatMax', 'Trackbars')
        val_min = cv2.getTrackbarPos('ValMin', 'Trackbars')
        val_max = cv2.getTrackbarPos('ValMax', 'Trackbars')
        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        return mask
    
    def threshold(self, mask):
        _, thresh_img = cv2.threshold(
            mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return thresh_img
    
    def find_contours(self, thresh_img):
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def two_largest_contours(self, contours):
        if len(contours) < 2:
            return contours  # Return all contours if fewer than 2 exist
        sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:2]
        approx_contours = []
        for cnt in sorted_contours:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)
        return approx_contours

    def clean_image(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_eroded = cv2.erode(mask, kernel, iterations=1)
        img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
        return img_dilated

    def centroid(self, contour):
        if len(contour) == 0:
            return (-1, -1)
        M = cv2.moments(contour)
        try:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        except ZeroDivisionError:
            return (-1, -1)
        return (x, y)
    
    def get_centroid(self,frame):
        mask = self.create_mask(frame)
        clean_mask = self.clean_image(mask)
        thresh_img = self.threshold(clean_mask)
        contours = self.find_contours(thresh_img)
        largest_contours = self.two_largest_contours(contours)
        centroids = []
        for contour in largest_contours:
            cx, cy = self.centroid(contour)
            centroids.append((cx, cy))
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        return centroids
        

################# DRIVER CODE ###################

# hd = HandDetection()
# hd.create_trackbars()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     mask = hd.create_mask(frame)
#     clean_mask = hd.clean_image(mask)
#     thresh_img = hd.threshold(clean_mask)
#     contours = hd.find_contours(thresh_img)
#     largest_contours = hd.two_largest_contours(contours)

#     for contour in largest_contours:
#         cx, cy = hd.centroid(contour)
#         cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()