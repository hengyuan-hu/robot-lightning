from . import cameras
from .robot import RobotEnv
try:
    from .spacemouse import SpaceMouseInterface
except ImportError:
    print("[robots] SpaceMouseInterface not loaded. Likely missing spacemouse driver.")

try:
    from .controllers.remote import ZeroRPCClient, ZeroRPCServer
except ImportError:
    print("[robots] ZeroRPC controllers not loaded.")

try:
    from .controllers.polymetis import PolyMetisController
except ImportError:
    print("[robots] ZeroRPC controllers not loaded.")

try:
    from .vr import VRController
except ImportError:
    print("[robots] VRController not loaded. oculus_reader likely missing.")
