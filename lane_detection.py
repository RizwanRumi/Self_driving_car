import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection(object):
    """docstring for LaneDetection"""
    def __init__(self):
        pass

    """docstring for image: bgr to gray conversion"""
    def channel_conversion(self, img):
        conv_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return conv_img

    """docstring for image noise reduction"""
    def feature_extraction(self, img):
        # image blurrring, parameters need to tune
        blur = cv2.GaussianBlur(img, (5,5), 0)

        # edge detection by canny
        canny = cv2.Canny(blur, 50, 150)
        return canny

    """docstring for ROI"""
    def region_of_interest(self, img):

        height = img.shape[0]
        width = img.shape[1]
        #print(width, height)

        region_of_interest_vertices = [
            (0, height), (width / 2, height / 2), (width, height)
        ]

        triangle =  np.array([region_of_interest_vertices], np.int32)

        masked_image = self.mask_image(triangle, img)

        roi = cv2.bitwise_and(img, masked_image)
        return roi

    """docstring for roi masking"""
    def mask_image(self, shape, img):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, shape, 255)
        return mask

    """docstring for hough transform"""
    def hough_transform(self, img):
        lines = cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        return lines

    def display_line(self, img, lines):
        line_image = np.zeros_like(img)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return line_image


    def line_coordinates(self, img, line_parameteres):
        line_slope = line_parameteres[0]
        line_intercept = line_parameteres[1]

        y1 = img.shape[0]
        y2 = int(y1 * (3 / 5))

        x1 = int((y1 - line_intercept) / line_slope)
        x2 = int((y2 - line_intercept) / line_slope)
        return np.array([x1, y1, x2, y2])


    def line_average_slope_intercept(self, img, lines):
        left_fit = []
        right_fit = []

        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope, intercept))

        if len(left_fit) and len(right_fit):
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)

            left_line = self.line_coordinates(img, left_fit_average)
            right_line = self.line_coordinates(img, right_fit_average)

            #print(left_line)
            #print(right_line)
            return np.array([left_line, right_line])



    """docstring for lane detection"""
    def lane_detection(self, img):
        # step 2 channel conversion
        copy_img = np.copy(img)
        gray = self.channel_conversion(copy_img)
        # step 3 noise reduction
        canny = self.feature_extraction(gray)
        # step 4 find ROI
        cropped_image = self.region_of_interest(canny)
        # step 5 apply Hough transform
        lines = self.hough_transform(cropped_image)
        # print(lines)
        line_average = self.line_average_slope_intercept(copy_img, lines)
        #line_image = self.display_line(copy_img, lines)
        line_image = self.display_line(copy_img, line_average)

        lane = cv2.addWeighted(img, 0.8, line_image, 1, 0)

        return lane

def feed_from_video(detection):
    cap = cv2.VideoCapture('videos/Lane.mp4')

    frame_count = 1

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frame)

    while total_frame > frame_count:
    #while cap.isOpened():
        print(frame_count)
        frame_count += 1

        ret, frame = cap.read()
        lane = detection.lane_detection(frame)

        cv2.imshow("lane detection", lane)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if(frame_count == total_frame):
            frame_count = 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    cap.release()
    cv2.destroyAllWindows()


def feed_from_image(detection):
    # step 1 read image
    img = cv2.imread("images/test_image.jpg")
    img = cv2.resize(img, (320, 320))
    # step 6 display line
    lane = detection.lane_detection(img)

    # plt.imshow(line_image)
    # plt.show()

    cv2.imshow("lane detection", lane)
    cv2.waitKey(0)


if __name__ == "__main__":

    lane_detection = LaneDetection()
    #feed_from_image(lane_detection)
    feed_from_video(lane_detection)


