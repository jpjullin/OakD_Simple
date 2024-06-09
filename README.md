# Oak-D Camera - Simple class to get frames

This class provides an easy-to-use interface for capturing frames from the Oak-D camera. 
It also offers simplified access to various camera parameters.

## Features
- Frame Capture: Get frames from the Oak-D camera
- Image Warping: Apply warping to the frames (onboard)
- OSC Integration: Use OSC (Open Sound Control) messages to show or hide frames and warp the image
- Mesh Saving: Save mesh data as json file

## Requirements
Clone the repository and install the necessary dependencies:
```
git clone https://github.com/jpjullin/OakD_Simple
cd OakD_Simple
pip install -r requirements.txt
```

## Usage Example

```
from OakD_Simple import OakD, cv2

# Initialize the OakD camera
oakd = OakD()

# Start the camera
oakd.run()

# Display frames in a loop
while True:
    if oakd.frame is not None:
        cv2.imshow("Frame", oakd.frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
```

## Class Initialization Arguments
```
oakd = OakD(
    # USB type
    usb_type='usb3',  # Options: usb2 | usb3
    
    # OSC port
    osc_port=2223,

    # Camera parameters
    cam='right',  # Options: right | left
    res="720",  # Options: 800 | 720 | 400
    fps=30,  # Frame/s (mono cameras)
    show_frame=False,  # Show the output frame (+fps)

    # Night vision
    laser_val=0,  # Project dots for active depth (0 to 1)
    ir_val=1,  # IR Brightness (0 to 1)

    # Image processing
    q_size=8,   # More = more latency, but smoother

    # Verbose
    verbose=False,  # Print (some) info about cam
)
```
