import cv2, numpy

def camera_capture(idx: int, *, window:str = "camera_capture") -> numpy.ndarray:
    """Preview a camera, and retrieve a frame from it."""
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {idx}")
    while True:
        r, f = cap.read()
        if not r:
            print("Camera capture interrupted and/or dropped!")
            break
        cv2.imshow(window, f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow(window)
    return f

def display_frame(f: numpy.ndarray, *, window:str = "preview") -> None:
    """Display a single frame as a preview."""
    while True:
        cv2.imshow(window, f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window)


