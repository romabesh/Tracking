import cv2 as cv

camera = cv.VideoCapture("video/Cars - 1900.mp4")
camera_detect = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=35)

while(camera.isOpened()):
    # если ошибка то ret = True
    ret, frame = camera.read()

    mask = camera_detect.apply(frame)
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            #cv.drawContours(frame, [cnt], contourIdx=-1, color=[0, 255, 255])
            x, y, h, w = cv.boundingRect(cnt)
            #print(x, y, h, w)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255))

    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()




