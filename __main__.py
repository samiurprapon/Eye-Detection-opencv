import cv2

# get camera capture in realtime
video = cv2.VideoCapture(0)

# face and eye Cascade model 
face_cascade = cv2.CascadeClassifier("haarcascade_face.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:

    # get video window and frames per second
    window, frame = video.read()

    # convert the image into gray scale to detect object in a lower compututaional power 
    # entire_area = cv2.cvtColor(source, color_code)
    entire_area = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces using cascade model inside the selected area 
    # this time entire_area is the whole window
    faces = face_cascade.detectMultiScale(entire_area, 1.3, 5)

    for (x, y, w, h) in faces:

        # draw rectange at FACE
        # cv2.rectangle(img, start_point, end_point, color, thickness) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # narrow down the face from entire window
        detected_face_area = entire_area[y:y+h, x:x+w]

        # select a frame for draw rectange later
        eye_frame = frame[y:y+h, x:x+w]
        
        # detect eyes inside of face rather finding all over the window
        eyes = eye_cascade.detectMultiScale(detected_face_area)

        for (ex, ey, ew, eh) in eyes:
            # draw rectange at EYES
            # cv2.rectangle(img, start_point, end_point, color, thickness) 
            cv2.rectangle(eye_frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Displaying the video
        # cv2.imshow(window_title, source)
        # mirrored the video with -> cv2.flip(source, vertically/horizontally = 0/1)
        cv2.imshow("Detection", cv2.flip(frame, 1))

    # wait for user input in every 1ms
    # checks the unnicode value of q using ord() and compare
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()