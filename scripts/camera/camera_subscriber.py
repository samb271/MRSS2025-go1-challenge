import zmq
import json
import numpy as np
import time
from threading import Thread, Lock
import queue


class ZMQCameraSubscriber:
    def __init__(self, server_ip="localhost", zmq_port=5555, timeout_ms=1000):
        self.server_ip = server_ip
        self.zmq_port = zmq_port
        self.timeout_ms = timeout_ms

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{server_ip}:{zmq_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        # Threading
        self.running = False
        self.latest_frame = None
        self.latest_metadata = None
        self.frame_lock = Lock()
        self.receive_thread = None

        # Statistics
        self.frames_received = 0
        self.last_frame_time = 0

        print(f"ZMQ Camera Subscriber connected to {server_ip}:{zmq_port}")

    def _receive_loop(self):
        """Background thread to receive frames"""
        while self.running:
            try:
                # Receive metadata
                metadata_json = self.socket.recv_string(zmq.NOBLOCK)
                metadata = json.loads(metadata_json)

                # Receive image data
                image_data = self.socket.recv(zmq.NOBLOCK)

                # Decode image based on encoding
                if metadata["encoding"] == "jpeg":
                    # For navigation without OpenCV, you might want raw data
                    # But if you can import cv2 in navigation env, use this:
                    try:
                        import cv2

                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except ImportError:
                        # If no cv2, store raw JPEG data
                        frame = image_data
                else:
                    # Handle other encodings if needed
                    frame = image_data

                # Update latest frame thread-safely
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_metadata = metadata
                    self.frames_received += 1
                    self.last_frame_time = time.time()

            except zmq.Again:
                # Timeout - no message received
                continue
            except Exception as e:
                if self.running:
                    print(f"Error receiving frame: {e}")
                time.sleep(0.01)

    def start(self):
        """Start receiving frames in background"""
        if self.running:
            return

        self.running = True
        self.receive_thread = Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        print("Camera subscriber started")

    def stop(self):
        """Stop receiving frames"""
        self.running = False

        if self.receive_thread:
            self.receive_thread.join(timeout=2)

        self.socket.close()
        self.context.term()

        print(f"Camera subscriber stopped. Received {self.frames_received} frames total.")

    def get_latest_frame(self):
        """
        Get the most recent frame and its metadata.

        Returns:
            tuple: (frame, metadata) or (None, None) if no frame available
        """
        with self.frame_lock:
            return self.latest_frame, self.latest_metadata

    def get_frame_blocking(self, timeout=1.0):
        """
        Wait for a new frame (blocking call).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            tuple: (frame, metadata) or (None, None) if timeout
        """
        start_time = time.time()
        initial_count = self.frames_received

        while time.time() - start_time < timeout:
            if self.frames_received > initial_count:
                return self.get_latest_frame()
            time.sleep(0.001)

        return None, None

    def is_receiving(self):
        """Check if frames are being received recently"""
        if self.last_frame_time == 0:
            return False
        return time.time() - self.last_frame_time < 2.0  # 2 second threshold

    def get_stats(self):
        """Get reception statistics"""
        return {
            "frames_received": self.frames_received,
            "last_frame_time": self.last_frame_time,
            "is_receiving": self.is_receiving(),
            "latest_metadata": self.latest_metadata,
        }


class NavigationCamera:
    """
    Simplified interface for navigation code.
    This class provides OpenCV-like interface using ZMQ frames.
    """

    def __init__(self, server_ip="localhost", zmq_port=5555):
        self.subscriber = ZMQCameraSubscriber(server_ip, zmq_port)
        self.subscriber.start()

        # Wait for first frame
        print("Waiting for camera connection...")
        frame, metadata = self.subscriber.get_frame_blocking(timeout=5.0)
        if frame is not None:
            print(f"Camera connected! Resolution: {metadata['width']}x{metadata['height']}")
        else:
            print("Warning: No frames received from camera")

    def read(self):
        """
        OpenCV-like interface: returns (ret, frame)

        Returns:
            tuple: (success_bool, frame_array) similar to cv2.VideoCapture.read()
        """
        frame, metadata = self.subscriber.get_latest_frame()

        if frame is not None:
            # If frame is raw JPEG data (no cv2 available), you'll need to handle this
            # For now, assuming frame is already decoded numpy array
            return True, frame
        else:
            return False, None

    def get_frame_with_metadata(self):
        """Get frame with full metadata for navigation algorithms"""
        return self.subscriber.get_latest_frame()

    def is_opened(self):
        """Check if camera is receiving frames"""
        return self.subscriber.is_receiving()

    def release(self):
        """Release camera resources"""
        self.subscriber.stop()


# Example usage functions
def simple_navigation_example():
    """Example of how to use in navigation code"""

    # Initialize camera
    camera = NavigationCamera()

    if not camera.is_opened():
        print("Failed to connect to camera")
        return

    try:
        frame_count = 0
        while True:
            # Get frame (similar to cv2.VideoCapture)
            ret, frame = camera.read()

            if ret:
                frame_count += 1

                # Your navigation processing here
                # process_frame_for_navigation(frame)

                if frame_count % 30 == 0:
                    print(f"Processing frame {frame_count} for navigation")

                # Example: get frame with metadata
                frame, metadata = camera.get_frame_with_metadata()
                if metadata:
                    print(f"Frame timestamp: {metadata['timestamp']}")

            time.sleep(0.033)  # ~30 FPS processing

    except KeyboardInterrupt:
        print("Navigation stopped by user")
    finally:
        camera.release()


def raw_subscriber_example():
    """Example of direct subscriber usage"""

    subscriber = ZMQCameraSubscriber()
    subscriber.start()

    try:
        while True:
            frame, metadata = subscriber.get_frame_blocking(timeout=1.0)

            if frame is not None:
                print(f"Received frame {metadata['frame_count']} from camera {metadata['cam_id']}")

                # Process frame for navigation
                # your_navigation_algorithm(frame)
            else:
                print("No frame received (timeout)")

    except KeyboardInterrupt:
        print("Stopping subscriber")
    finally:
        subscriber.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZeroMQ Camera Subscriber")
    parser.add_argument("--server", default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ port")
    parser.add_argument("--mode", choices=["simple", "raw"], default="simple", help="Example mode to run")

    args = parser.parse_args()

    print(f"=== ZeroMQ Camera Subscriber ===")
    print(f"Server: {args.server}:{args.port}")
    print(f"Mode: {args.mode}")

    if args.mode == "simple":
        simple_navigation_example()
    else:
        raw_subscriber_example()
