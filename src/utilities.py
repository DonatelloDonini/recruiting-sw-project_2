import cv2
import numpy as np

def add_info_panel(frame, parameters: dict, direction: str, internal: bool=False, bg_color=(0, 0, 0), border_thickness=1, border_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_color=(255, 255, 255)):
    """
    Create an information panel and attach it to a given frame.

    Parameters:
        frame (numpy.ndarray): The image frame on which to overlay the info panel.
        parameters (dict): A dictionary of parameters to display on the info panel.
        direction (str): The direction to place the info panel relative to the frame.
                        Options are 'top', 'bottom', 'left', 'right'.
        internal (bool, optional): If True, overlay the info panel inside the frame.
                                If False, stack the info panel outside the frame. Default is False.
        bg_color (tuple, optional): Background color of the info panel in BGR format. Default is (0, 0, 0).
        border_thickness (int, optional): Thickness of the border around the info panel. Default is 1.
        border_color (tuple, optional): Color of the border around the info panel in BGR format. Default is (255, 255, 255).
        font (int, optional): Font type for the text. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_size (float, optional): Font size for the text. Default is 0.5.
        font_color (tuple, optional): Color of the text in BGR format. Default is (255, 255, 255).

    Returns:
        numpy.ndarray: The frame with the info panel overlay.
    """
    # Get the size of the frame
    frame_height, frame_width = frame.shape[:2]

    # Calculate the size of the info panel based on the number of parameters and frame size
    padding= 40
    max_text_size= max([cv2.getTextSize(f"{key}: {value}", font, font_size, 1)[0][0] for key, value in parameters.items()])
    line_height = 20
    panel_height = padding + line_height * len(parameters)
    panel_width = max_text_size+ int(padding/2)

    if direction in ("left", "right"):
        panel_height= frame_height
    if direction in ("top", "bottom"):
        panel_width= frame_width

    # Create the info panel as a black rectangle with specified background color
    info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    info_panel[:, :] = bg_color

    # Add the parameter text to the info panel
    for i, (key, value) in enumerate(parameters.items()):
        param_text = f"{key}{value}"
        cv2.putText(info_panel, param_text, (10, 20 + line_height * i), font, font_size, font_color, 1, cv2.LINE_AA)

    if internal:
        # Overlay the info panel on the frame (position at top-left corner by default)
        if direction == 'bottom':
            frame[frame_height - panel_height:frame_height, :panel_width] = info_panel
        elif direction == 'left':
            frame[:panel_height, :panel_width] = info_panel
        elif direction == 'right':
            frame[:panel_height, frame_width - panel_width:frame_width] = info_panel
        else:
            frame[:panel_height, :panel_width] = info_panel
    else:
        # Stack the frame and the info panel based on the specified direction
        if direction == 'bottom':
            frame = np.vstack((frame, info_panel))
        elif direction == 'left':
            frame = np.hstack((info_panel, frame))
        elif direction == 'right':
            frame = np.hstack((frame, info_panel))
        else:
            frame = np.vstack((info_panel, frame))

    return frame

def create_canvas(width, height, bg_color):
    """
    Create a canvas of given width, height, and background color.

    Args:
        width (int): Width of the canvas in pixels.
        height (int): Height of the canvas in pixels.
        bg_color (tuple): Background color as (B, G, R) in the range 0-255.

    Returns:
        canvas (numpy.ndarray): The created canvas image.
    """
    # Create a blank image using the provided dimensions and color
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = bg_color
    return canvas