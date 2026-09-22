"""HID driver for SpaceMouse devices with split or combined motion reports.

Install ``hidapi`` (not the incompatible ``hid`` package) and connect the device
before starting. On Linux, a udev rule must grant access to its USB/HID node.
Vendor/product IDs are auto-detected when the configured IDs cannot be opened.
"""

import threading
import time
from collections import namedtuple

import numpy as np
from pynput.keyboard import Listener

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

try:
    import hid
except ModuleNotFoundError as exc:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. Install it with `pip install hidapi`."
    ) from exc


import robosuite.macros as macros
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse(Device):
    """
    A minimalistic driver class for SpaceMouse with HID library.
    Auto-detects 3Dconnexion devices if default vendor/product IDs fail.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        vendor_id (int): SpaceMouse vendor ID (will auto-detect if default fails)
        product_id (int): SpaceMouse product ID (will auto-detect if default fails)
        device_path (bytes): Specific device path (e.g., b'1-2:1.0') for reliable identification
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(
        self,
        env,
        vendor_id=macros.SPACEMOUSE_VENDOR_ID,
        product_id=macros.SPACEMOUSE_PRODUCT_ID,
        device_path=None,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
    ):
        super().__init__(env)

        ROBOSUITE_DEFAULT_LOGGER.info("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()

        if device_path:
            try:
                self.device.open_path(device_path)
                ROBOSUITE_DEFAULT_LOGGER.info(f"Connected using path: {device_path}")
            except OSError:
                ROBOSUITE_DEFAULT_LOGGER.warning(f"Failed to open device at path: {device_path}")
                self._auto_detect_device()
        else:
            try:
                self.device.open(vendor_id, product_id)
                ROBOSUITE_DEFAULT_LOGGER.info(f"Connected using default IDs: {vendor_id:04x}:{product_id:04x}")
            except OSError:
                ROBOSUITE_DEFAULT_LOGGER.warning(
                    f"Failed to open device with provided IDs: {vendor_id:04x}:{product_id:04x}"
                )
                self._auto_detect_device()

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        ROBOSUITE_DEFAULT_LOGGER.info("Manufacturer: %s" % self.device.get_manufacturer_string())
        ROBOSUITE_DEFAULT_LOGGER.info("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False
        self._pressed_keys = set()
        self._keyboard_grasp = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        try:
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        except BaseException:
            self.device.close()
            raise
        self._stop_event = threading.Event()
        self._read_error = None
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # start listening
        try:
            self.listener.start()
        except BaseException:
            self.close()
            raise

    def _auto_detect_device(self):
        """Auto-detect and connect to first 3Dconnexion device."""
        devices = [
            d
            for d in hid.enumerate()
            if d.get("vendor_id") == 0x256F
            or (d.get("vendor_id") == 0x046D and d.get("product_id") in {0xC626, 0xC627, 0xC628, 0xC62B})
            or d.get("manufacturer_string") == "3Dconnexion"
        ]
        if not devices:
            raise OSError("No 3Dconnexion devices found")

        selected = devices[0]
        try:
            self.device.open_path(selected["path"])
        except OSError as exc:
            self.device.close()
            raise OSError(
                "SpaceMouse found but cannot be opened. On Linux, grant this user access to the "
                "device with a udev rule for USB vendor/product "
                f"{selected['vendor_id']:04x}:{selected['product_id']:04x}, then reconnect it. "
                "Also close any other application using the device."
            ) from exc
        self.vendor_id = selected["vendor_id"]
        self.product_id = selected["product_id"]
        ROBOSUITE_DEFAULT_LOGGER.info(f"Auto-detected: {selected['product_string']} with path {selected['path']}")

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("R / right button", "retry trajectory")
        print_command("Space", "toggle gripper")
        print_command("W/A/S/D", "move arm horizontally")
        print_command("Q/E", "move arm down/up")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("Control+C", "quit")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")
        print("NOTE: Auto-detects 3Dconnexion devices. Use device_path for specific device.")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False
        self._pressed_keys = set()
        self._keyboard_grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        if self._read_error is not None:
            raise OSError("SpaceMouse HID reader failed") from self._read_error
        dpos = (
            np.array(
                [
                    ("d" in self._pressed_keys) - ("a" in self._pressed_keys),
                    ("w" in self._pressed_keys) - ("s" in self._pressed_keys),
                    ("e" in self._pressed_keys) - ("q" in self._pressed_keys),
                ],
                dtype=float,
            )
            * 0.005
            * self.pos_sensitivity
        )
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        while not self._stop_event.is_set():
            try:
                report = self.device.read(13, timeout_ms=100)
            except OSError as exc:
                self._read_error = exc
                return
            if report and self._enabled:
                self._process_report(report)

    def _process_report(self, report):
        """Decode both split 7-byte and combined 13-byte motion reports."""
        if report[0] == 1 and len(report) >= 7:
            self.y = convert(report[1], report[2])
            self.x = convert(report[3], report[4])
            self.z = -convert(report[5], report[6])
            if len(report) >= 13:
                self.roll = convert(report[7], report[8])
                self.pitch = convert(report[9], report[10])
                self.yaw = convert(report[11], report[12])
        elif report[0] == 2 and len(report) >= 7:
            self.roll = convert(report[1], report[2])
            self.pitch = convert(report[3], report[4])
            self.yaw = convert(report[5], report[6])
        elif report[0] == 3 and len(report) >= 2:
            self.single_click_and_hold = bool(report[1] & 1)
            if report[1] & 2:
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    def close(self):
        """Stop listeners before releasing the HID handle; safe to call twice."""
        self._enabled = False
        self._stop_event.set()
        self.listener.stop()
        self.thread.join()
        if self.device is not None:
            self.device.close()
            self.device = None

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold or self._keyboard_grasp:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        name = getattr(key, "char", None)
        name = "space" if name == " " else name
        if name not in {"w", "a", "s", "d", "q", "e", "r", "space"}:
            return
        first_press = name not in self._pressed_keys
        self._pressed_keys.add(name)
        if name == "space" and first_press:
            self._keyboard_grasp = not self._keyboard_grasp
        elif name == "r" and first_press:
            self._reset_state = 1
            self._enabled = False

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        name = getattr(key, "char", None)
        self._pressed_keys.discard("space" if name == " " else name)

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(100):
        ROBOSUITE_DEFAULT_LOGGER.info(f"Control: {space_mouse.control}, Gripper: {space_mouse.control_gripper}")
        time.sleep(0.02)
