from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from threading import Thread
from pathlib import Path
import depthai as dai
import numpy as np
import time
import json
import cv2


class OakD:
    def __init__(self, usb_type='usb3', osc_port=2223, cam='right', res="720", fps=30,
                 laser_val=0, ir_val=1, show_frame=False, q_size=8, verbose=False):
        self.device = None
        self.pipeline = None

        self.q_rectified = None
        self.q_warped = None
        self.q_max_size = q_size

        self.frame_rectified = None
        self.frame_warped = None
        self.frame = None

        self.running = False
        self.restart_device = False
        self.previous_time = None
        self.windows = set()

        # USB type
        self.usb_type = usb_type  # Options: usb2 | usb3

        # OSC
        self.osc_receive_ip = "0.0.0.0"
        self.osc_receive_port = osc_port
        self.server = None

        # Main parameters
        self.resolution = res  # Options: 800 | 720 | 400
        self.fps = fps  # Frame/s (mono cameras)
        self.show_frame = show_frame  # Show the output frame (+fps)
        self.stream = cam  # Options: right | left

        # Night vision
        self.laser_val = laser_val  # Project dots for active depth (0 to 1)
        self.ir_val = ir_val  # IR Brightness (0 to 1)
        self.tuning = 'tuning_mono_low_light.bin'  # Low light tuning file
        self.tuning = Path(__file__).parent.joinpath(self.tuning)

        # Stereo parameters
        self.lrcheck = True  # Better handling for occlusions
        self.extended = False  # Closer-in minimum depth, disparity range is doubled
        self.subpixel = True  # Better accuracy for longer distance
        self.median = "7x7"  # Options: OFF | 3x3 | 5x5 | 7x7

        # Verbose
        self.verbose = verbose  # Print (some) info about cam

        # Mesh
        self.mesh_file = 'mesh.json'
        self.mesh_file = Path(__file__).parent.joinpath(self.mesh_file)

        # Resolution
        self.res_map = {
            '800': {'w': 1280, 'h': 800, 'res': dai.MonoCameraProperties.SensorResolution.THE_800_P},
            '720': {'w': 1280, 'h': 720, 'res': dai.MonoCameraProperties.SensorResolution.THE_720_P},
            '400': {'w': 640, 'h': 400, 'res': dai.MonoCameraProperties.SensorResolution.THE_400_P}
        }
        self.resolution = self.res_map[self.resolution]

        # Median kernel
        self.median_map = {
            "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
            "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
            "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
            "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
        }
        self.median = self.median_map[self.median]

        # Warp
        self.warp_pos = self.load_custom_mesh()

        # Init device
        self.init_device()

    def init_device(self):
        self.pipeline = dai.Pipeline()

        # Low light tuning
        self.pipeline.setCameraTuningBlobPath(self.tuning)

        # Mono cameras
        cam_left = self.pipeline.create(dai.node.MonoCamera)
        cam_right = self.pipeline.create(dai.node.MonoCamera)
        cam_left.setCamera("left")
        cam_right.setCamera("right")

        # Set resolution and fps
        for mono_cam in (cam_left, cam_right):
            mono_cam.setResolution(self.resolution['res'])
            mono_cam.setFps(self.fps)

        # Create stereo pipeline
        stereo = self.pipeline.create(dai.node.StereoDepth)
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        # Stereo settings
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(self.median)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setLeftRightCheck(self.lrcheck)
        stereo.setExtendedDisparity(self.extended)
        stereo.setSubpixel(self.subpixel)

        # Stream out rectified
        xout_rectif = self.pipeline.create(dai.node.XLinkOut)
        xout_rectif.setStreamName("rectified")

        # Create warp pipeline
        warp = self.pipeline.create(dai.node.Warp)

        # Linking
        if self.stream == "right":
            stereo.rectifiedRight.link(xout_rectif.input)
            stereo.rectifiedRight.link(warp.inputImage)
        else:
            stereo.rectifiedLeft.link(xout_rectif.input)
            stereo.rectifiedLeft.link(warp.inputImage)

        # Warp settings
        warp.setWarpMesh(self.warp_pos, 2, 2)
        warp.setOutputSize(self.resolution['w'], self.resolution['h'])
        warp.setMaxOutputFrameSize(self.resolution['w'] * self.resolution['h'] * 3)
        warp.setHwIds([0])
        warp.setInterpolation(dai.Interpolation.NEAREST_NEIGHBOR)

        # Stream out warped
        xout_warped = self.pipeline.create(dai.node.XLinkOut)
        xout_warped.setStreamName("warped")
        warp.out.link(xout_warped.input)

        # Initialize device
        if self.usb_type == 'usb2':
            self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)
        else:
            self.device = dai.Device(self.pipeline)

        # Verbose
        if self.verbose:
            self.device.setLogLevel(dai.LogLevel.DEBUG)
            self.device.setLogOutputLevel(dai.LogLevel.DEBUG)

        # Queues
        self.q_rectified = self.device.getOutputQueue(name="rectified", maxSize=self.q_max_size, blocking=False)
        self.q_warped = self.device.getOutputQueue(name="warped", maxSize=self.q_max_size, blocking=False)
        self.previous_time = time.time()

        # OSC
        self.init_osc()

    def init_osc(self):
        disp = Dispatcher()
        disp.map("/*", lambda osc_address, *msg: self.handle_msg(osc_address, msg))
        self.server = ThreadingOSCUDPServer((self.osc_receive_ip, self.osc_receive_port), disp)
        print("Listening on port", self.osc_receive_port)
        osc_thread = Thread(target=self.server.serve_forever, daemon=True)
        osc_thread.start()

    def handle_msg(self, osc_address, msg):
        address_handlers = {
            "/show_frame": lambda: setattr(self, 'show_frame', bool(msg[0])),
            "/warp_pos": lambda: setattr(self, 'warp_pos', [
                (
                    int(msg[i * 2] * self.resolution['w']),
                    int(msg[(i * 2) + 1] * self.resolution['h'])
                )
                if i * 2 < len(msg) and (i * 2) + 1 < len(msg)
                else (0, 0)
                for i in range(4)
            ]),
            "/warp_go": lambda: (
                print("Mesh changed, restarting..."),
                setattr(self, 'restart_device', True),
            ),
            "/save": lambda: self.save_mesh(),
        }
        handler = address_handlers.get(osc_address)
        if handler:
            handler()

    def create_mesh(self):
        coordinates = []
        for i in range(2):
            for j in range(2):
                x = j * self.resolution['w']
                y = i * self.resolution['h']
                coordinates.append((x, y))
        return coordinates

    def save_mesh(self):
        if not isinstance(self.warp_pos, list):
            self.warp_pos = self.warp_pos.tolist()
        with open(self.mesh_file, 'w') as filehandle:
            json.dump(self.warp_pos, filehandle)
        print('Mesh saved to', self.mesh_file)

    def load_custom_mesh(self):
        if self.mesh_file.is_file():
            with open(str(self.mesh_file), 'r') as data:
                mesh = json.loads(data.read())
            print("Custom mesh loaded")
            return np.array(mesh)
        else:
            print("No custom mesh")
            return self.create_mesh()

    @staticmethod
    def create_gui():
        window_name = "Oak-D Tracking"
        width = 200
        height = 100

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        gui_bg = np.ones((height, width, 3), np.uint8) * 255
        cv2.rectangle(gui_bg, (0, 0), (width, height), (0, 0, 0), -1)

        gui_text = "Press q to stop"
        gui_font = cv2.FONT_HERSHEY_SIMPLEX
        gui_font_scale = 0.7
        gui_font_thickness = 1
        gui_text_size = cv2.getTextSize(gui_text, gui_font, gui_font_scale, gui_font_thickness)[0]

        # Center text
        gui_text_x = (width - gui_text_size[0]) // 2
        gui_text_y = (height + gui_text_size[1]) // 2
        cv2.putText(gui_bg, gui_text, (gui_text_x, gui_text_y), gui_font, gui_font_scale,
                    (255, 255, 255), gui_font_thickness, cv2.LINE_AA)

        return gui_bg

    def _run_thread(self):
        cv2.namedWindow("Oak-D Tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("Oak-D Tracking", OakD.create_gui())

        while self.running:
            self.frame_rectified = self.q_rectified.get().getCvFrame()

            start_time = time.time()
            self.frame_warped = self.q_warped.get().getCvFrame()
            end_time = time.time()

            print('Time to get frame:', end_time - start_time)

            self.frame = self.frame_warped

            if self.show_frame:
                # Show source frame (without warping)
                if self.frame_rectified is not None:
                    source = cv2.cvtColor(self.frame_rectified, cv2.COLOR_GRAY2BGR)
                    color = (0, 0, 255)
                    for i in range(4):
                        cv2.circle(source, (int(self.warp_pos[i][0]), int(self.warp_pos[i][1])), 4, color, -1)
                        if i % 2 != 2 - 1:
                            cv2.line(source,
                                     (int(self.warp_pos[i][0]), int(self.warp_pos[i][1])),
                                     (int(self.warp_pos[i + 1][0]), int(self.warp_pos[i + 1][1])),
                                     color, 2)
                        if i + 2 < 4:
                            cv2.line(source,
                                     (int(self.warp_pos[i][0]), int(self.warp_pos[i][1])),
                                     (int(self.warp_pos[i + 2][0]), int(self.warp_pos[i + 2][1])),
                                     color, 2)
                    cv2.imshow("Source", source)
                    self.windows.add("Source")

                # Show warped frame
                if self.frame_warped is not None:
                    current_time = time.time()
                    fps = 1 / (current_time - self.previous_time)
                    self.previous_time = current_time
                    frame = cv2.cvtColor(self.frame_warped, cv2.COLOR_GRAY2BGR)
                    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    cv2.imshow("Warped", frame)
                    self.windows.add("Warped")

            else:
                for window in self.windows:
                    cv2.destroyWindow(window)
                self.windows.clear()

            if self.restart_device:
                self.stop()
                self.init_device()
                self.run()
                self.restart_device = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        self.running = True
        run_thread = Thread(target=self._run_thread)
        run_thread.start()

    def stop(self):
        self.running = False
        if self.device is not None:
            self.device.close()
            self.device = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
