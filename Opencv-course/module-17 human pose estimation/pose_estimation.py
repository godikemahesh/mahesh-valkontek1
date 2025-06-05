import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
cap=cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,10)
#cap.set(3,640)
#cap.set(4,640)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)

while cv2.waitKey(1)<0:
    has_img,image=cap.read()
    image_height, image_width, _ = image.shape

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect body parts
    results = pose.process(image_rgb)


    # Function to apply transformations
    def transform_coordinates(coordinates, translation=(0, 0), rotation=0, scale=1):
        # Translation
        transformed = coordinates + np.array(translation)

        # Rotation
        angle = np.deg2rad(rotation)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        transformed = np.dot(transformed, rotation_matrix)

        # Scaling
        transformed *= scale

        return transformed


    # Extract and transform keypoints
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            keypoints.append([x, y])

        keypoints = np.array(keypoints)

        # Example transformation: Translate right by 50 pixels and up by 30 pixels
        transformed_keypoints = transform_coordinates(keypoints, translation=(40,-30))

        # Draw the transformed keypoints on the image
        for point in transformed_keypoints:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)


        cv2.imshow('Transformed Image',image)




cv2.waitKey(0)
cv2.destroyAllWindows()






