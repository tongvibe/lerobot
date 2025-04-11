# SO100 Real Robot

#how to use it
#(lerobot) tong@tong-3090:~/lerobot$ python3 packages/openpi-client/lerobot_clinet.py --use_policy
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# Import tqdm for progress bar
from tqdm import tqdm

# Imports for OpenPI Client
from openpi_client import image_tools
from openpi_client import websocket_client_policy

#################################################################################
# SO100Robot Class (Keep as is, no changes needed here)
#################################################################################
class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False, camera_indices=(9,10,11)):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.camera_indices = camera_indices
        self.top_camera_index = camera_indices[0]
        self.third_camera_index = camera_indices[1]
        self.wrist_camera_index = camera_indices[2]
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "top": OpenCVCameraConfig(self.top_camera_index, 30, 640, 480, "bgr"),
                "third": OpenCVCameraConfig(self.third_camera_index, 30, 640, 480, "bgr"), # Third camera config
                "wrist": OpenCVCameraConfig(self.wrist_camera_index, 30, 640, 480, "bgr"),
            }
        self.config.leader_arms = {}

        if self.calibrate:
            import os
            import shutil
            # Construct the absolute path starting from the user's home directory
            home_dir = os.path.expanduser("~") # Gets '/home/tong'
            absolute_calibration_path_base = os.path.join(home_dir, "lerobot", ".cache", "calibration")
            calibration_folder = os.path.join(absolute_calibration_path_base, "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]
        self.top_camera = None
        self.third_camera = None
        self.wrist_camera = None

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )
        self.motor_bus.connect()
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.robot.activate_calibration()
        self.set_so100_robot_preset()
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        if self.enable_camera:
            self.top_camera = self.robot.cameras["top"]
            self.third_camera = self.robot.cameras["third"] # Assign third camera
            self.wrist_camera = self.robot.cameras["wrist"]
            self.top_camera.connect()
            self.third_camera.connect() # Connect third camera
            self.wrist_camera.connect()
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        self.motor_bus.write("Mode", 0)
        self.motor_bus.write("P_Coefficient", 10)
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        self.motor_bus.write("Lock", 0)
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state", current_state)
        print("observation keys:", self.robot.capture_observation().keys())
        # current_state[0] = 90
        # current_state[2] = 90
        # current_state[3] = 90
        # self.robot.send_action(current_state)
        # time.sleep(2)
        # current_state[4] = -70
        # current_state[5] = 30
        # current_state[1] = 90
        # self.robot.send_action(current_state)
        # time.sleep(2)
        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        print("----------------> SO100 Robot moved to home pose")
        # home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        # self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        # Ensure the state is returned as a NumPy array, which it should be already
        # If not, convert: return self.get_observation()["observation.state"].data.numpy()
        # Also ensure dtype is suitable for JSON serialization if needed, float/int is fine.
        return self.get_observation()["observation.state"].data.numpy().astype(np.float64)


    def get_current_img(self, camera_name="top"):
        if camera_name not in ["top", "third", "wrist"]:
            raise ValueError("Invalid camera name. Must be 'top', 'third', or 'wrist'.")
        camera_key = f"observation.images.{camera_name}"
        # Get raw BGR image from camera
        img_bgr = self.get_observation()[camera_key].data.numpy()
        # Convert to RGB for consistency and potential use with viewers/other libraries
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb # Return RGB uint8 image

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        # Check if connected before trying to disable/disconnect
        if self.robot.is_connected:
            try:
                self.disable()
            except Exception as e:
                 print(f"Warning: Error disabling motors during disconnect: {e}")
            self.robot.disconnect()
            self.robot.is_connected = False
            if self.enable_camera:
                # Check if camera objects exist before disconnecting
                if self.top_camera: self.top_camera.disconnect()
                if self.third_camera: self.third_camera.disconnect() # Disconnect third camera
                if self.wrist_camera: self.wrist_camera.disconnect()
            print("================> SO100 Robot disconnected")
        else:
            print("================> SO100 Robot already disconnected")


    def __del__(self):
        # Ensure disconnection happens even if context manager isn't used properly
        self.disconnect()

#################################################################################
# NEW OpenPI Client Adapter (Modified for Third Camera)
#################################################################################
class OpenPIClientAdapter:
    def __init__(
        self,
        host="localhost",
        port=8000, # Default OpenPI port
        language_instruction="Grasp red, green, yellow ducks and put them in the box.",
        resize_size=224, # Typical resize size for OpenPI models
    ):
        self.language_instruction = language_instruction
        self.resize_size = resize_size
        # Initialize the OpenPI websocket client
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        print(f"Attempting to connect to OpenPI policy server at ws://{host}:{port}")
        # Optional: Add a check or wait for connection if the client library supports it
        # Or handle connection errors during the first `get_action` call.

    def get_action(self, top_img_rgb, third_img_rgb, wrist_img_rgb, state): # Added third_img_rgb
        """
        Prepares observations and gets action chunk from the OpenPI policy server.

        Args:
            top_img_rgb: NumPy array (H, W, 3), RGB uint8 format from the top camera.
            third_img_rgb: NumPy array (H, W, 3), RGB uint8 format from the third camera.
            wrist_img_rgb: NumPy array (H, W, 3), RGB uint8 format from the wrist camera.
            state: NumPy array (6,) representing the robot state (joints + gripper).  <-- Should be NumPy array

        Returns:
            NumPy array: Action chunk of shape (action_horizon, action_dim).
        """
        # 1. Preprocess images: Resize with padding and convert to uint8
        if top_img_rgb.dtype != np.uint8:
            print("Warning: top_img_rgb is not uint8, converting.")
            top_img_rgb = top_img_rgb.astype(np.uint8)
        if third_img_rgb.dtype != np.uint8:
            print("Warning: third_img_rgb is not uint8, converting.")
            third_img_rgb = third_img_rgb.astype(np.uint8)
        if wrist_img_rgb.dtype != np.uint8:
            print("Warning: wrist_img_rgb is not uint8, converting.")
            wrist_img_rgb = wrist_img_rgb.astype(np.uint8)

        processed_top_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(top_img_rgb, self.resize_size, self.resize_size)
        )
        processed_third_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(third_img_rgb, self.resize_size, self.resize_size)
        )
        processed_wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img_rgb, self.resize_size, self.resize_size)
        )

        # 2. Construct the observation dictionary
        # --- Ensure 'state' is passed as a NumPy array ---
        if not isinstance(state, np.ndarray):
            print(f"Warning: state received in get_action is not a NumPy array (type: {type(state)}). Converting.")
            state = np.array(state, dtype=np.float64) # Or appropriate dtype

        observation = {
            "observation/image": processed_top_img,
            "observation/third": processed_third_img,
            "observation/wrist_image": processed_wrist_img,
            # --- Send state as NumPy array ---
            "observation/state": state,
            "prompt": self.language_instruction,
        }

        # 3. Call the policy server
        start_time = time.time()
        try:
            # The client library should handle serializing numpy arrays (usually to lists in JSON)
            result = self.client.infer(observation)
            print(f"Inference query time taken: {time.time() - start_time:.4f}s")
        except Exception as e:
            print(f"Error during inference call to OpenPI server: {e}")
            raise e # Re-raise the exception

        # 4. Extract and return the action chunk
        action_chunk = result.get("actions")
        if action_chunk is None:
             raise ValueError("Policy server response did not contain 'actions' key.")
        if not isinstance(action_chunk, np.ndarray):
             action_chunk = np.array(action_chunk, dtype=np.float32)

        return action_chunk
    # Optional: Add a close method if the client library requires explicit closing
    # def close(self):
    #     self.client.close() # If such a method exists

#################################################################################
# Matplotlib Viewer (Keep as is)
#################################################################################
def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame

#################################################################################
# Main Execution Block (Modified for Third Camera)
#################################################################################
if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true", help="Use OpenPI policy server instead of dataset playback."
    )
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    # Default host/port for OpenPI server
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host IP.")
    parser.add_argument("--port", type=int, default=6006, help="Policy server port.") # Default port 6006 for OpenPI server
    parser.add_argument("--action_horizon", type=int, default=8, help="Number of actions to execute from the returned chunk.") # Adjust default based on policy
    parser.add_argument("--actions_to_execute", type=int, default=350, help="Total number of policy inference calls.")
    # --- Updated default camera indices ---
    parser.add_argument("--camera_index_top", type=int, default=12) # Example index
    parser.add_argument("--camera_index_third", type=int, default=4) # Example index
    parser.add_argument("--camera_index_wrist", type=int, default=6) # Example index
    # --- End updated camera indices ---
    parser.add_argument("--instruction", type=str, default="Pick up the fruits and place them on the plate.", help="Task instruction for the policy.")
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute # Number of inference calls
    USE_POLICY = args.use_policy
    # Client-side horizon: How many steps to execute from each fetched action chunk
    CLIENT_EXECUTION_HORIZON = args.action_horizon
    # Note: The policy server might return a chunk of a different length. We execute the first CLIENT_EXECUTION_HORIZON steps.


    if USE_POLICY:
        print("Using OpenPI Policy Server...")
        # Instantiate the NEW client adapter
        client = OpenPIClientAdapter(
            host=args.host,
            port=args.port,
            language_instruction=args.instruction,
        )

        # Instantiate the robot with potentially updated camera indices
        robot = SO100Robot(calibrate=False, enable_camera=True, camera_indices=(args.camera_index_top, args.camera_index_third, args.camera_index_wrist))

        plt.figure("Realtime View") # Create a figure for the viewer

        try: # Use try/finally to ensure robot disconnects
            with robot.activate():
                print("Robot activated. Starting policy execution loop...")
                for _ in tqdm(range(ACTIONS_TO_EXECUTE), desc="Policy Execution Steps"):
                    # 1. Get current observations from robot
                    top_img_rgb = robot.get_current_img(camera_name="top")
                    third_img_rgb = robot.get_current_img(camera_name="third") # <-- Now getting third image
                    wrist_img_rgb = robot.get_current_img(camera_name="wrist")
                    current_state = robot.get_current_state()

                    # Display the top camera view (original resolution)
                    # You could modify view_img or add another plt.subplot to show the third image too
                    view_img(top_img_rgb)

                    # 2. Get action chunk from policy server
                    try:
                         # --- Updated call to include third_img_rgb ---
                         action_chunk = client.get_action(top_img_rgb, third_img_rgb, wrist_img_rgb, current_state)
                         # --- End updated call ---

                         # Returned chunk shape: (server_horizon, action_dim)
                         # action_dim should be 6 for SO100
                         if action_chunk.shape[1] != 6:
                             print(f"Warning: Expected action dimension 6, but got {action_chunk.shape[1]}.")
                             # Decide how to handle this - skip, error, etc.
                             # continue # Skip this chunk maybe?

                    except Exception as e:
                        print(f"Failed to get action from policy server: {e}")
                        print("Stopping execution.")
                        break # Exit the loop on policy error

                    # 3. Execute actions from the chunk
                    num_actions_in_chunk = action_chunk.shape[0]
                    steps_to_execute = min(CLIENT_EXECUTION_HORIZON, num_actions_in_chunk)

                    #print(f"Received action chunk shape: {action_chunk.shape}. Executing {steps_to_execute} steps.")

                    inner_loop_start_time = time.time()
                    for i in range(steps_to_execute):
                        # Get the action for the current step
                        action_for_step_i = action_chunk[i]

                        # Ensure it's a numpy array before converting to tensor
                        if not isinstance(action_for_step_i, np.ndarray):
                             action_for_step_i = np.array(action_for_step_i, dtype=np.float32)

                        # Check shape just in case
                        if action_for_step_i.shape != (6,):
                             print(f"Error: Action at step {i} has incorrect shape {action_for_step_i.shape}. Expected (6,). Skipping remaining chunk.")
                             break # Stop executing this chunk

                        # Send action to robot
                        robot.set_target_state(torch.from_numpy(action_for_step_i).float())

                        # Short sleep between sending commands
                        time.sleep(0.02) # Adjust sleep time as needed

                        # Optional: Get and display real-time image *during* execution
                        # current_top_img = robot.get_current_img(camera_name="top")
                        # view_img(current_top_img) # This will slow down execution significantly

                    #print(f"Executed {steps_to_execute} actions from chunk in {time.time() - inner_loop_start_time:.4f}s")

        except KeyboardInterrupt:
            print("Execution interrupted by user.")
        except Exception as e:
             print(f"An error occurred during robot operation: {e}")
        finally:
            print("Disconnecting robot...")
            plt.close("Realtime View") # Close the viewer window
            # The robot disconnect is handled by the __del__ and context manager exit
            # but calling it explicitly here ensures it happens before program exit
            # if an error occurred outside the 'with' block or if __del__ fails.
            if 'robot' in locals() and robot.robot.is_connected:
                 robot.disconnect()
            # Optional: Close policy client if needed
            # if 'client' in locals() and hasattr(client, 'close'):
            #     client.close()
            print("Cleanup complete.")

    else:
        # Test Dataset Playback (Keep as is)
        print("Using Dataset Playback...")
        dataset = LeRobotDataset(
            repo_id="Loki0929/so100_duck", # Or use args.dataset_path if it's a local path
            root=args.dataset_path, # Specify root if repo_id is used for download location
        )

        # Instantiate the robot with potentially updated camera indices
        robot = SO100Robot(calibrate=False, enable_camera=True, camera_indices=(args.camera_index_top, args.camera_index_third, args.camera_index_wrist))

        plt.figure("Dataset Playback") # Create a figure for the viewer

        try: # Use try/finally for disconnect
            with robot.activate():
                actions_from_dataset = []
                num_frames = min(ACTIONS_TO_EXECUTE, len(dataset)) # Ensure we don't exceed dataset length
                print(f"Loading {num_frames} actions from dataset...")

                for i in tqdm(range(num_frames), desc="Loading Actions"):
                    data_item = dataset[i]
                    action = data_item["action"] # Assuming 'action' key holds the target state tensor
                    actions_from_dataset.append(action)

                    # Optionally view dataset image vs realtime image during loading
                    # img_dataset_chw = data_item["observation.images.webcam"].data.numpy() # Assuming 'webcam' is top
                    # img_dataset_hwc = img_dataset_chw.transpose(1, 2, 0)
                    # realtime_img_rgb = robot.get_current_img(camera_name="top")
                    # view_img(img_dataset_hwc, realtime_img_rgb) # Overlay

                print("Executing loaded actions...")
                for action in tqdm(actions_from_dataset, desc="Executing Actions"):
                    # Get current image for viewing (only top shown by default)
                    realtime_img_rgb = robot.get_current_img(camera_name="top")
                    # You could get third/wrist images here too if needed for comparison or viewing
                    # realtime_third_img_rgb = robot.get_current_img(camera_name="third")
                    # realtime_wrist_img_rgb = robot.get_current_img(camera_name="wrist")
                    view_img(realtime_img_rgb)

                    # Send action (should be a Tensor already from LeRobotDataset)
                    if not isinstance(action, torch.Tensor):
                        action = torch.from_numpy(action).float() # Convert if necessary

                    robot.set_target_state(action)
                    time.sleep(0.05) # Original sleep time for dataset playback

                print("Finished executing dataset actions.")
                robot.go_home()
                print("Robot moved to home position.")

        except KeyboardInterrupt:
            print("Execution interrupted by user.")
        except Exception as e:
             print(f"An error occurred during dataset playback: {e}")
        finally:
            print("Disconnecting robot...")
            plt.close("Dataset Playback")
            if 'robot' in locals() and robot.robot.is_connected:
                 robot.disconnect()
            print("Cleanup complete.")