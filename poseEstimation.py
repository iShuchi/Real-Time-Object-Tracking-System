import cv2
import cv2.aruco as aruco
import numpy as np

# Displaying menu to user
print("Choose an ArUco dictionary:")
print("1: 4X4_1000")
print("2: 5X5_1000")
print("3: 6X6_1000")
print("4: 7X7_1000")

user_choice = input("Enter the number corresponding to your choice: ")

user_choice = int(user_choice)

cap = cv2.VideoCapture(0)

# Camera Calibration
fx = 1000.0 
fy = 1000.0 
cx = 320.0 
cy = 240.0   

cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

while True:
    _, img = cap.read()

    # Press 'e' to exit the Window
    if cv2.waitKey(1) == ord('e'):  
        break

    # Adding Markers
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if user_choice == 1:
        chosen_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    elif user_choice == 2:
        chosen_dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    elif user_choice == 3:
        chosen_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    elif user_choice == 4:
        chosen_dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    else:
        print("Invalid choice. Run Code again & choose option 1/2/3/4.")
        break
        
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, chosen_dictionary, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)

        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]                             # rvec = rotational vector, tvec = translational vector

            axis_points, _ = cv2.projectPoints(np.float32([[0, 0, 0], [0, 0.1, 0], [0.1, 0, 0], [0, 0, -0.1]]), rvec, tvec, cameraMatrix, distCoeffs)

            axis_points = np.int32(axis_points).reshape(-1, 2)

            img = cv2.line(img, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 3)  # X-axis (red)
            img = cv2.line(img, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 3)  # Y-axis (green)
            img = cv2.line(img, tuple(axis_points[0]), tuple(axis_points[3]), (255, 0, 0), 3)  # Z-axis (blue)

    cv2.imshow('Display', img)

cap.release()
cv2.destroyAllWindows()