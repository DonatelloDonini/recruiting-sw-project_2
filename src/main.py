import cv2
import numpy as np
import time
import csv

import constants
from settings import Settings
from utilities import add_info_panel, create_canvas

DEBUG= True

def extract_keypoints(frame, grayscale_mask_dimensions, canny_thresholds, epsilon_scalar, min_points_density) -> list:
    # Grayscaling
    grayscale= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the edges
    if len(grayscale_mask_dimensions)== 1:
        grayscale_mask_dimensions= (grayscale_mask_dimensions[0], grayscale_mask_dimensions[0])

    blurred = cv2.GaussianBlur(grayscale, grayscale_mask_dimensions, 0)

    if DEBUG:
        cv2.imshow("Blurred", blurred)
        cv2.moveWindow("Blurred", 0, 0)

    # Canny edge detection
    edged = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        cv2.imshow("Edged", edged)
        cv2.moveWindow("Edged", 510, 0)

    # Extract the corners from the contours
    keypoints= []
    for contour in contours:
        epsilon = epsilon_scalar * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= min_points_density:
            cluster_x_sum= 0
            cluster_y_sum= 0

            for point in approx:
                cluster_x_sum+= point[0][0]
                cluster_y_sum+= point[0][1]

            cluster_average_point= (cluster_x_sum// len(approx), cluster_y_sum// len(approx))

            keypoints.append(cluster_average_point)

    return keypoints

def get_close_keypoints(point, points_to_check, radius, shape= constants.SQUARE):

    close_points= []

    # Calculate the bounding box of the shape
    left = point[0] - radius
    right = point[0] + radius
    top = point[1] - radius
    bottom = point[1] + radius

    if (constants.SQUARE== shape):
        for point_to_check in points_to_check:
            if left <= point_to_check[0] <= right and top <= point_to_check[1] <= bottom:
                close_points.append(point_to_check)

    elif (constants.CIRCLE== shape):
        for point_to_check in points_to_check:
            if (
                (left <= point_to_check[0] <= right and top <= point_to_check[1] <= bottom) and # point fits within the bounding box
                ((point[0] - point_to_check[0]) ** 2 + (point[1] - point_to_check[1]) ** 2) <= radius ** 2 # point fits within the circle
                ):
                close_points.append(point_to_check)

    else:
        raise ValueError("Invalid shape.")

    return close_points

def get_pixel_surrounding_area(point, radius, frame, shape= constants.SQUARE) -> list:

    nearby_pixels= []
    frame_height= frame.shape[0]
    frame_width= frame.shape[1]

    # Calculate the bounding box of the shape
    left = max(0, (point[0] - radius))
    top = max(0, point[1] - radius)
    right = min(frame_width, point[0] + radius)
    bottom = min(frame_height, point[1] + radius)

    if (constants.SQUARE== shape):
        for x in range(left, right):
            for y in range(top, bottom):
                nearby_pixels.append(frame[y, x])

    elif (constants.CIRCLE== shape):
        for x in range(left, right):
            for y in range(top, bottom):
                if ((point[0] - x) ** 2 + (point[1] - y) ** 2) <= radius ** 2:
                    nearby_pixels.append(frame[y, x])
    else:
        raise ValueError("Invalid shape.")

    return nearby_pixels

def get_similarity_coefficient(area1, area2) -> int:
    if len(area1)< len(area2):
        area2= area2[:len(area1)]
    elif len(area1)> len(area2):
        area1= area1[:len(area2)]

    total_difference= 0
    for pixel1, pixel2 in zip(area1, area2):
        total_difference+= abs(int(pixel1)- int(pixel2))

    return round(total_difference / len(area1))


def get_best_matching_point(point, close_points, prev_frame, curr_frame, surrounding_check_area_radius, color_space= constants.GRAYSCALE, shape= constants.SQUARE):

    best_match= None
    current_min_differece= None

    if (constants.GRAYSCALE== color_space):
        curr_frame= cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_frame= cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    elif (constants.BGR== color_space):
        None
    else:
        raise ValueError("The colors space is invalid")

    pixels_surrounding_point= get_pixel_surrounding_area(point, surrounding_check_area_radius, prev_frame, shape)
    for close_point in close_points:
        pixels_surrounding_close_point= get_pixel_surrounding_area(close_point, surrounding_check_area_radius, curr_frame, shape)
        difference= get_similarity_coefficient(pixels_surrounding_point, pixels_surrounding_close_point)

        if (current_min_differece is None) or (difference< current_min_differece):
            best_match= close_point
            current_min_differece= difference

    return best_match


def bind_keypoints(prev_frame, curr_frame, grayscale_mask_dimensions, canny_thresholds, epsilon_scalar, min_points_density, interframe_check_radius, surrounding_check_area_radius= None, color_space= constants.GRAYSCALE, interframe_check_shape= constants.SQUARE):
    prev_keypoints= extract_keypoints(prev_frame, grayscale_mask_dimensions, canny_thresholds, epsilon_scalar, min_points_density)
    curr_keypoints= extract_keypoints(curr_frame, grayscale_mask_dimensions, canny_thresholds, epsilon_scalar, min_points_density)

    if DEBUG:
        curr_frame_copy= curr_frame.copy()
        for keypoint in prev_keypoints:
            cv2.circle(curr_frame_copy, tuple(keypoint), 2, (0, 0, 255), -1)
            if (constants.SQUARE== interframe_check_shape):
                cv2.rectangle(curr_frame_copy, (keypoint[0] - interframe_check_radius, keypoint[1] - interframe_check_radius), (keypoint[0] + interframe_check_radius, keypoint[1] + interframe_check_radius), (0, 0, 255), 1)
            elif (constants.CIRCLE== interframe_check_shape):
                cv2.circle(curr_frame_copy, tuple(keypoint), interframe_check_radius, (0, 0, 255), 1)

        for keypoint in curr_keypoints:
            cv2.circle(curr_frame_copy, tuple(keypoint), 2, (0, 255, 0), -1)

        cv2.imshow("ExtractedKeypoints", curr_frame_copy)
        cv2.moveWindow("ExtractedKeypoints", 510*2, 0)

    # Find the keypoints from one frame to the other that are close to each other
    close_keypoints= {}
    for prev_keypoint in prev_keypoints:
        close_keypoints[tuple(prev_keypoint)]= get_close_keypoints(prev_keypoint, curr_keypoints, interframe_check_radius, interframe_check_shape)

    # Find the best matching keypoint from the close ones
    matches= []
    for prev_keypoint, close_keypoints in close_keypoints.items():
        if len(close_keypoints)> 0:
            surrounding_check_area_radius= surrounding_check_area_radius if (surrounding_check_area_radius is not None) else interframe_check_radius
            best_match= get_best_matching_point(
                prev_keypoint,
                close_keypoints,
                prev_frame,
                curr_frame,
                surrounding_check_area_radius,
                color_space,
                interframe_check_shape
            )
            matches.append((prev_keypoint, best_match))

    return matches

def linear_heat_dispersion(canvas, point, radius, start_value):
    """
    Draws a gray fading circle on the provided canvas.

    The circle starts with the specified grayscale value at the center and fades towards white (255) at the perimeter.

    Args:
        canvas (numpy.ndarray): The canvas where the circle is drawn.
        point (tuple): The center of the circle (x, y).
        radius (int): The radius of the circle.
        start_value (int): The starting grayscale value (0-255) at the center of the circle.
    """

    canvas_width= canvas.shape[1]
    canvas_height= canvas.shape[0]

    # Calculating circle bounding box
    left = max(0, point[0] - radius)
    right = min(canvas_width, point[0] + radius)
    top = max(0, point[1] - radius)
    bottom = min(canvas_height, point[1] + radius)

    fading_function= lambda distance: int((-start_value/radius) * distance + start_value)

    for x in range(left, right):
        for y in range(top, bottom):
            distance= np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)

            if distance <= radius:
                canvas[y, x] = min(255, canvas[y, x][0]+ fading_function(distance))

def gaussian_heat_dispersion(canvas, point, radius, start_value, sigma= None):
    """
    Draws a gray fading circle with a Gaussian gradient on the provided canvas.

    The circle starts with black (or the specified value) at the center and fades to white towards the edge.

    Args:
        canvas (numpy.ndarray): The canvas where the circle is drawn.
        point (tuple): The center of the circle (x, y).
        radius (int): The radius of the circle.
        start_value (int): The starting grayscale value (0-255) at the perimeter of the circle.
        sigma (float): The standard deviation of the Gaussian function controlling the spread of the fade.
    """

    canvas_width= canvas.shape[1]
    canvas_height= canvas.shape[0]

    # Calculating circle bounding box
    left = max(0, point[0] - radius)
    right = min(canvas_width, point[0] + radius)
    top = max(0, point[1] - radius)
    bottom = min(canvas_height, point[1] + radius)

    if sigma is None:
        sigma= radius/2

    fading_function= lambda distance: int(start_value * np.exp(-(distance ** 2) / (2 * sigma ** 2)))

    for x in range(left, right):
        for y in range(top, bottom):
            distance= np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)

            if distance <= radius:
                canvas[y, x] = min(255, canvas[y, x][0]+ fading_function(distance))

def update_heatmap(keypoints_heatmap, bound_keypoints, keypoint_temperature: int, heat_dispersion_radius: int, cooldown_coefficient: int, heat_dispersion_behaviour: int= constants.LINEAR):

    # Take the first channel for efficiency reasons (I know the image is grayscale)
    gray_channel = keypoints_heatmap[:, :, 0]
    sub_value = np.full(gray_channel.shape, cooldown_coefficient, dtype=np.uint8)
    output_gray = cv2.subtract(gray_channel, sub_value)

    keypoints_heatmap = cv2.merge([output_gray, output_gray, output_gray])

    for point_couple in bound_keypoints:
        for point in point_couple:
            if (constants.LINEAR== heat_dispersion_behaviour):
                linear_heat_dispersion(keypoints_heatmap, point, heat_dispersion_radius, keypoint_temperature)
            elif (constants.GAUSSIAN== heat_dispersion_behaviour):
                gaussian_heat_dispersion(keypoints_heatmap, point, heat_dispersion_radius, keypoint_temperature)
            else:
                raise ValueError("Invalid heat dispersion behaviour.")

    return keypoints_heatmap

def heatmap_filter(keypoints: list, heatmap, threshold: int)-> list:
    valid_keypoints= []
    for keypoint in keypoints:
        if (((0<= keypoint[0]) and (keypoint[0]< heatmap.shape[1])) and ((0<= keypoint[1]) and (keypoint[1]< heatmap.shape[0]))) and heatmap[keypoint[1], keypoint[0]][0]< threshold:
            valid_keypoints.append(keypoint)

    return valid_keypoints


def calculate_average_angle(point_pairs):
    angles = []

    for pair in point_pairs:
        # Extract start and end points
        (x1, y1), (x2, y2) = pair

        # Calculate the vector angle using atan2
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    # Calculate the average angle
    average_angle = np.mean(angles)

    return average_angle

def main():
    ###                          ###
    ### Code to extract settings ###
    ###                          ###

    settings= Settings("settings.json")

    video_path= settings.get("video_path")
    frame_width= settings.get("frame_width")
    frame_height= settings.get("frame_height")
    grayscale_mask_dimensions= settings.get("grayscale_mask_dimensions")
    canny_thresholds= settings.get("canny_thresholds")
    epsilon_scalar= settings.get("epsilon_scalar")
    min_points_density= settings.get("min_points_density")
    interframe_check_radius= settings.get("interframe_check_radius")
    surrounding_check_area_radius= settings.get("surrounding_check_area_radius")
    color_space= settings.get("color_space")
    interframe_check_shape= settings.get("interframe_check_shape")
    keypoints_temperature= settings.get("keypoints_temperature")
    heat_dispersion_radius= settings.get("heat_dispersion_radius")
    cooldown_coefficient= settings.get("cooldown_coefficient")
    heat_dispersion_behaviour= settings.get("heat_dispersion_behaviour")
    movement_threshold= settings.get("movement_threshold")
    heat_threshold= settings.get("heat_threshold")

    ###                      ###
    ### Stats initialization ###
    ###                      ###

    homography_file_path = "./statistics/homography.csv"
    homography_file= None
    csv_writer= None

    try:
        homography_file= open(homography_file_path, "w", newline="")

        csv_writer= csv.DictWriter(homography_file, fieldnames=[
            "frame_index",
            "homography_matrix",
            "found_keypoints",
            "valid_keypoints",
            "non_valid_keypoints",
            "standard_deviation",
            "average_difference_from_mean",
            "vector_angle_radians"
        ])
        csv_writer.writeheader()

        if DEBUG:
            print(f"File '{homography_file_path}' created successfully.")

    except FileExistsError:
        if DEBUG:
            print(f"File '{homography_file_path}' already exists.")

    ###                                   ###
    ### Code to initialize the video loop ###
    ###                                   ###

    cap= cv2.VideoCapture(video_path)

    frame_index= 0
    prev_frame= None
    IS_PAUSED= False

    keypoints_heatmap= None

    while True:
        ret, frame= cap.read()

        if not ret:
            break

        ###                   ###
        ### Scaling the frame ###
        ###                   ###

        if frame_width and not frame_height:
            frame_height = int(frame.shape[0] * frame_width / frame.shape[1])
        elif frame_height and not frame_width:
            frame_width = int(frame.shape[1] * frame_height / frame.shape[0])
        elif not frame_width and not frame_height:
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

        frame= cv2.resize(frame, (frame_width, frame_height))

        if (keypoints_heatmap is None):
            keypoints_heatmap= create_canvas(frame_width, frame_height, (0, 0, 0))

        if (frame_index== 0):
            prev_frame= frame
            frame_index+= 1
            continue

        frame_start_time = time.time()

        ###                       ###
        ### Binding the keypoints ###
        ###                       ###

        bound_keypoints= None
        if prev_frame is not None:
            bound_keypoints= bind_keypoints(
                prev_frame,
                frame,
                grayscale_mask_dimensions,
                canny_thresholds,
                epsilon_scalar,
                min_points_density,
                interframe_check_radius,
                surrounding_check_area_radius,
                color_space,
                interframe_check_shape
            )

            heat_dispersion_radius= interframe_check_radius if (heat_dispersion_radius is None) else heat_dispersion_radius
            keypoints_heatmap= update_heatmap(
                keypoints_heatmap,
                bound_keypoints,
                keypoints_temperature,
                heat_dispersion_radius,
                cooldown_coefficient,
                heat_dispersion_behaviour
            )

        ###                   ###
        ### Showing the frame ###
        ###                   ###

        # Drawing interframe movement onto the frame
        valid_keypoints_count= 0
        valid_bound_keypoints= []
        for prev_keypoint, curr_keypoint in bound_keypoints:
            has_moved_enough= (curr_keypoint[0] - prev_keypoint[0]) ** 2 + (curr_keypoint[1] - prev_keypoint[1]) ** 2 > movement_threshold**2

            ###                                        ###
            ### Filtering out the platform's keypoints ###
            ###                                        ###

            valid_keypoints= heatmap_filter([prev_keypoint, curr_keypoint], keypoints_heatmap, heat_threshold)

            if has_moved_enough and len(valid_keypoints)== 2:
                cv2.arrowedLine(frame, (prev_keypoint[0], prev_keypoint[1]), (curr_keypoint[0], curr_keypoint[1]), (255, 0, 0), 1, tipLength= .5)
                valid_bound_keypoints.append((prev_keypoint, curr_keypoint))
                valid_keypoints_count+= 1
            elif DEBUG:
                cv2.arrowedLine(frame, (prev_keypoint[0], prev_keypoint[1]), (curr_keypoint[0], curr_keypoint[1]), (0, 0, 255), 1, tipLength= .5)

        ###                       ###
        ### Generating statistics ###
        ###                       ###

        frame_processing_time = time.time() - frame_start_time
        fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0

        starting_points= np.array([keyopint_binding[0] for keyopint_binding in valid_bound_keypoints])
        ending_points= np.array([keyopint_binding[1] for keyopint_binding in valid_bound_keypoints])

        homography_matrix= None
        if (len(starting_points)> 4) and (len(ending_points)> 4):
            homography_matrix, _ = cv2.findHomography(starting_points, ending_points)

        ###              ###
        ### Info display ###
        ###              ###

        parameters_to_display= {
            "OpenCV version: ": cv2.__version__,
            "FPS: ": round(fps, 2),
            "FRAME INDEX: ": frame_index,
            "KEYPOINTS FOUND: ": len(bound_keypoints) if bound_keypoints is not None else 0,
            "VALID KEYPOINTS: ": valid_keypoints_count,
            "DEBUG VIEW: ": DEBUG
        }

        frame= add_info_panel(frame, parameters_to_display, "bottom")

        cv2.imshow("Result", frame)
        cv2.moveWindow("Result", 510, frame_height+ 30)

        if DEBUG:
            cv2.imshow("KeypointsHeatmap", cv2.applyColorMap(keypoints_heatmap, cv2.COLORMAP_JET))
            cv2.moveWindow("KeypointsHeatmap", 0, frame_height+ 30)

        ###                       ###
        ### Writing stats to file ###
        ###                       ###

        if csv_writer is not None:
            csv_writer.writerow({
                "frame_index": frame_index,
                "homography_matrix": homography_matrix,
                "found_keypoints": len(bound_keypoints) if bound_keypoints is not None else 0,
                "valid_keypoints": valid_keypoints_count,
                "non_valid_keypoints": len(bound_keypoints) - valid_keypoints_count if bound_keypoints is not None else 0,
                "standard_deviation": np.std(ending_points - starting_points),
                "average_difference_from_mean": np.mean(ending_points - starting_points),
                "vector_angle_radians": calculate_average_angle(valid_bound_keypoints)
            })

        ###                ###
        ### Video controls ###
        ###                ###

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Restart
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            continue

        # Pause
        elif key == ord('p') or IS_PAUSED:
            IS_PAUSED = True

        # Go to next frame if paused
        while IS_PAUSED:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('p'):
                IS_PAUSED = False
            elif key == ord(' '):
                break

        prev_frame= frame
        frame_index+= 1

    ###                     ###
    ### Releasing resources ###
    ###                     ###

    cap.release()
    cv2.destroyAllWindows()

    if homography_file is not None:
        homography_file.close()


if __name__ == '__main__':
    main()