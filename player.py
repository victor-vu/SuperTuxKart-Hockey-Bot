from os import path
import numpy as np
import torch
import torchvision
from PIL import Image
from image_agent.imodels import imodel1, imodel3
import sys, os, json
from collections import deque
from itertools import islice

print("Loaded")
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    # 'mps' if torch.backends.mps.is_available() else
    "cpu"
)

if device == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# currently not used - just info.   Actual resize is in lines 370 or 371
resize_to = [144, 192]  # model: imodel1
resize_to = [128, 288]  # model: imodel2

# Settings
MAX_PUCK_DEV = 0.8
FLAG_THR = 0.6
DRIFT_THR = 0.8
STEER_MULTIPLIER = 15
TARGET_VELOCITY = 15
MAX_VELOCITY = 20
RANGE_X = (-45, 45)
RANGE_Y = (-65, 65)
KICKOFF_EVENT_DURATION = 35
UNSTUCK_EVENT_DURATION = 70
SEARCH_STEPS = 20
DEBUG_AND_OVERLAY = False


overlay_data = {
    "player1_circle": (0, 0),
    "player2_circle": (0, 0),
    "player1_flag": False,
    "player2_flag": False,
    "player1_steer": 0.0,
    "player2_steer": 0.0,
    "player1_state": "KICKOFF",
    "player2_state": "KICKOFF",
    "player1_loc": (0, 0),
    "player2_loc": (0, 0),
    "player1_depth": "0",
    "player2_depth": "0",
    "player1_p_est": [0, 0],
    "player2_p_est": [0, 0],
}


class CustomCrop(object):
    def __init__(self, start_row, end_row):
        self.start_row = start_row
        self.end_row = end_row

    def __call__(self, img, *args):
        # Crop the image from start_row to end_row vertically
        return (F.crop(img, self.start_row, 0, self.end_row - self.start_row, img.width),) + args


def generate_overlay_data():
    global overlay_data
    return overlay_data


def norm(vector):
    return torch.linalg.norm(vector)


def distance(point1, point2):  # should normally be 2d points
    return norm(point1 - point2)


def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2


def player_id_to_team_id(id):
    if id == 0 or id == 2:  # i.e (id % 2==0)
        return 1
    else:
        return 2


def get_goal(team_id):
    # player 0 and 2 = team1
    # player 1 and 3 = team2
    # 'goal_line': [[[-10.449999809265137, 0.07000000029802322, -64.5],
    #                [10.449999809265137, 0.07000000029802322, -64.5]],
    #               [[10.460000038146973, 0.07000000029802322, 64.5],
    #                [-10.510000228881836, 0.07000000029802322, 64.5]]]
    if team_id == 1:  # could be wrong team
        return torch.tensor([0, 75.0])  # ([0, -64.5])  # what is x and y ?
    else:
        return torch.tensor([0, -75.0])  # ([0, +64.5])


def get_own_goal(team_id):
    if team_id != 1:
        return torch.tensor([0, +75.0])
    else:
        return torch.tensor([0, -75.0])


def get_size_of_puck():
    return 2.527893304824829  # constant


def get_height_of_puck():
    return 0.37  # assume it's constant


def in_goal(kart_center):
    x, y = kart_center
    if y > 64:
        return True  # right goal
    if y < -64:
        return True  # left  goal
    return False


def in_start_position(kart_center):
    x, y = kart_center
    if -10 < x < 10 and ((-58 < y < -50) or (50 < y < 58)):
        return True
    return False


def move_out_of_goal(kart_center):
    x, y = kart_center
    if y > 64 and x >= 0:
        return 1
    if y > 64 and x < 0:
        return -1
    if y < -64 and x >= 0:
        return -1
    if y < -64 and x < 0:
        return 1


def screen_to_world(screen_x, screen_y, depth, view, projection):
    # Prepare the screen coordinates in homogeneous form
    s = torch.tensor([screen_x, -screen_y, depth, 1])

    # Calculate inverses of the matrices
    P_inv = torch.linalg.inv(projection)
    V_inv = torch.linalg.inv(view)
    r = V_inv @ P_inv @ s
    return torch.tensor([r[0] / r[-1], r[1] / r[-1], r[2] / r[-1]])


def projection_to_image_cords(x, proj, view):
    # needs transposed proj and view !!!
    # return x,y
    # from HW5
    p = proj @ view @ x
    return torch.tensor([p[0] / p[-1], -p[1] / p[-1], p[2] / p[-1]])


# leftover from refactoring
def world_to_screen_old(world_point_4d, projection, view):
    screen_x_max = 400  # holdovers from trainings set creation
    screen_y_max = 300

    projection_delta = projection_to_image_cords(world_point_4d, projection, view)
    center_x = screen_x_max / 2
    center_y = screen_y_max / 2

    screen_x = center_x + projection_delta[0] * center_x
    screen_y = center_y + projection_delta[1] * center_y

    normalized_x = (screen_x / (screen_x_max)) * 2 - 1
    normalized_y = (screen_y / (screen_y_max)) * 2 - 1

    return normalized_y, normalized_x


# TODO: check if it leads to exactly te same results
def world_to_screen(world_point_4d, projection, view):
    p = projection @ view @ world_point_4d
    projection_to_image_cords = torch.tensor([p[0] / p[-1], -p[1] / p[-1], p[2] / p[-1]])
    return projection_to_image_cords[1], projection_to_image_cords[0]  # normalized_y, normalized_x


def tensors_below_threshold(tensor_deque, threshold):
    for tensor in tensor_deque:
        if not torch.all(tensor < threshold):
            return False
    return True


class FIFOLocation:
    def __init__(self, buffer_size, thr=0.5):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # Using deque with a fixed maximum length
        self.location_thr = thr

    def add(self, tensor):
        # Deque will automatically discard the oldest item when exceeding the maxlen
        self.buffer.append(tensor)

    def stuck(self, location_thr=None):
        # Added optional parameter for threshold, defaulting to instance's location_thr
        if not self.buffer:
            return False
        location_thr = location_thr if location_thr is not None else self.location_thr
        stacked_tensors = torch.stack(list(self.buffer))
        differences = torch.abs(stacked_tensors[1:] - stacked_tensors[:-1])
        stuck = torch.mean(differences) < location_thr
        return stuck.item()


def get_player_info(player_num, own_state):
    # player num = 0 or 1
    kart_info = own_state[player_num]["kart"]
    camera_info = own_state[player_num]["camera"]

    velocity = torch.norm(torch.tensor(kart_info["velocity"], dtype=torch.float32)[[0, 2]])
    player_id = kart_info["player_id"]
    max_steer_angle = kart_info["max_steer_angle"]
    location = torch.tensor(kart_info["location"], dtype=torch.float32)[[0, 2]]  # center
    front = torch.tensor(kart_info["front"], dtype=torch.float32)[[0, 2]]
    projection = torch.tensor(camera_info["projection"].T)
    view = torch.tensor(camera_info["view"].T)

    return velocity, player_id, max_steer_angle, location, front, projection, view


def is_in_front_of_kart(kart_center, kart_front, puck_location, radian_thr=torch.tensor(np.pi) / 3):
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    powerup_direction = (puck_location - kart_center) / torch.norm(puck_location - kart_center)
    # Compute angle to power-up using dot product
    angle_to_powerup = torch.acos(torch.dot(kart_direction, powerup_direction))

    # Check if the power-up is within -pi/3 to pi/3 radians in front of the kart
    return -radian_thr <= angle_to_powerup <= radian_thr


def calculate_hit_point(player_center, puck_location, goal_center, puck_diameter):
    """
    Calculates the optimal hit point and the directional offset (left/right) for the player to hit the puck into the goal.

    Args:
    player_center (tuple): Coordinates of the player (x, y).
    puck_location (tuple): Coordinates of the puck (x, y).
    goal_center (tuple): Coordinates of the goal (x, y).
    puck_diameter (float): Diameter of the puck.

    Returns:
    tuple: Contains coordinates of the perfect hit point and the offset direction (negative for left, positive for right).
    """
    normalized_vector = (goal_center - puck_location) / torch.norm(goal_center - puck_location)
    # Scale the normalized vector by half the puck's diameter (radius)
    radius = puck_diameter / 2
    scaled_vector = normalized_vector * radius
    # Calculate the perfect hit point
    perfect_hit_point = puck_location - scaled_vector

    # Calculate vectors for distance calculation
    player_to_puck_vector = puck_location - player_center
    player_to_perfect_hit_point_vector = perfect_hit_point - player_center
    # Calculate perpendicular distance using cross product
    area = torch.cross(
        torch.cat([torch.tensor([1.0]), player_to_puck_vector], dim=0), torch.cat([torch.tensor([1.0]), player_to_perfect_hit_point_vector], dim=0)
    )[0]
    base_length = torch.norm(player_to_puck_vector)
    perpendicular_distance = area / base_length

    return (perfect_hit_point, -perpendicular_distance)


def toward_our_goal(kart_center, kart_front, point):
    # Compute the  direction vector of the kart and kart->goal
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    point_direction = (point - kart_center) / torch.norm(point - kart_center)

    # Calculate the angle between the current direction and the reverse of the to_goal direction
    angle_to_goal = torch.arctan2(point_direction[1], point_direction[0])
    angle_current = torch.arctan2(kart_direction[1], kart_direction[0])

    # Angle difference normalized between -pi to pi
    angle_difference = torch.arctan2(torch.sin(angle_to_goal - angle_current), torch.cos(angle_to_goal - angle_current))

    # Determine steering value based on the angular difference, normalize to [-1, 1]
    steering_value = -torch.sign(angle_difference) * torch.abs(angle_difference) / torch.tensor(np.pi)

    return steering_value


def steering_to_point(kart_center, kart_front, point):
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    point_direction = (point - kart_center) / torch.norm(point - kart_center)
    theta_kart_point = torch.acos(torch.dot(kart_direction, point_direction))
    theta_kart_point_left_or_right = -torch.sign(torch_2d_cross(kart_direction, point_direction))
    angle_kart_point_aim_offset = torch.rad2deg(theta_kart_point_left_or_right * theta_kart_point)
    return angle_kart_point_aim_offset, theta_kart_point_left_or_right


def torch_2d_cross(a, b):  # like np.cross
    a = torch.cat([torch.tensor([1]), a.squeeze()])
    b = torch.cat([torch.tensor([1]), b.squeeze()])
    r = torch.cross(a, b)
    return r[0]


def steer_to_point(kart_center, kart_front, puck_location, own_goal_center, reverse_thr=5):
    # print(kart_center, kart_front, puck_location, own_goal_center, reverse_thr)
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    puck_direction = (puck_location - kart_center) / torch.norm(puck_location - kart_center)

    kart_puck_distance = distance(kart_center, puck_location)
    in_front_180 = is_in_front_of_kart(kart_center, kart_front, puck_location, radian_thr=torch.tensor(np.pi) / 2)
    # in_front_120 = is_in_front_of_kart(kart_center, kart_front, puck_location, radian_thr=torch.tensor(np.pi) / 3)

    # dump steering calculation
    angle_to_puck = torch.atan2(puck_direction[1], puck_direction[0]) - torch.atan2(kart_direction[1], kart_direction[0])
    angle_to_puck = torch.remainder(angle_to_puck + torch.tensor(np.pi), 2 * torch.tensor(np.pi)) - torch.tensor(np.pi)  # Normalize angle to [-π, π]
    steering_direction = torch.sign(angle_to_puck)
    steering_value = -steering_direction * torch.abs(angle_to_puck) / torch.tensor(np.pi)

    # whenever we would thit the wall, use the opposite direction
    if kart_puck_distance > 15:
        return -steering_value.item(), False
    # if it's less than 5 I assume we can't get such a tight turning circle
    elif in_front_180 and kart_puck_distance > reverse_thr:
        return -steering_value.item(), True
    # else reverse toward our goal
    else:
        return toward_our_goal(kart_center, kart_front, own_goal_center), True


def check_if_onscreen(aim_point):
    x, y = aim_point[0], aim_point[1]
    if -1 <= x <= 1 and -1 <= y <= 1:
        return True
    else:
        return False


def distance_weighting_function(distance_goal):
    weight_between_1_and_2 = ((torch.clip(distance_goal, 10, 100) - 10) / 90) + 1
    weighting = 1 / 3 * (1 / weight_between_1_and_2**3)
    return weighting


class Team:
    agent_type = "image"

    def __init__(self):
        """
        TODO: Load your agent here. Load network parameters, and other parts of our model
        We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

        # try:
        # /Users/andreas/Desktop/UT_Austin/DeepLearning/FinalProject/image_agent/imodel3_144_192.th
        # # self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), "imodel.pt"))
        self.model = imodel1()
        print(path.join(path.dirname(path.abspath(__file__)), "imodel.th"))
        self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), "imodel.th"), map_location=torch.device(device)))
        # self.model = imodel3().to(device)
        # print(path.join(path.dirname(path.abspath(__file__)), "imodel3_144_192.th"))
        # print(device)
        # self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), "imodel3_144_192.th"), map_location=torch.device(device)))
        self.model.to(device)
        self.model.eval()
        # except Exception as e:
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        # print(e)

        self.location_fifo1 = FIFOLocation(15)
        self.location_fifo2 = FIFOLocation(15)
        # self.transform = torchvision.transforms.Compose([CustomCrop(92, 284), torchvision.transforms.Resize((128, 288)), torchvision.transforms.ToTensor()])
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((144, 192)), torchvision.transforms.ToTensor()])
        self.step = 0
        self.kickoff_steps_left = KICKOFF_EVENT_DURATION

        self.last_puck_screen_center1 = torch.tensor([0, 0])
        self.last_puck_screen_center2 = torch.tensor([0, 0])
        self.working_normaly1 = True
        self.working_normaly2 = True
        self.detection_history1 = deque(maxlen=4)
        self.location_history1 = deque(maxlen=4)
        self.last_positive_location1 = torch.tensor([0, 0])
        self.search_counter1 = 0
        self.reverse_timer1 = 0
        self.detection_history2 = deque(maxlen=4)
        self.location_history2 = deque(maxlen=4)
        self.velocity_history1 = deque(maxlen=12)
        self.velocity_history2 = deque(maxlen=12)

        self.last_positive_location2 = torch.tensor([0, 0])
        self.search_counter2 = 0
        self.reverse_timer2 = 0
        self.reversal_counter1 = 0
        self.reversal_counter2 = 0
        self.keep_tracking1 = 0
        self.keep_tracking2 = 0
        self.search_rear1 = 0
        self.search_rear2 = 0
        self.use_puck1 = True
        self.use_puck2 = True
        self.flag_prob_history1 = deque(maxlen=7)
        self.flag_prob_history2 = deque(maxlen=7)
        self.reversal_aim1 = torch.tensor([0, 1])
        self.reversal_aim2 = torch.tensor([0, 1])
        # print("INIT DONE")

    def confirm_detection(self, player_id=1):
        if player_id == 1:
            detection_history = list(self.detection_history1)[-3:]
            location_history = list(self.location_history1)[-3:]
        else:
            detection_history = list(self.detection_history2)[-3:]
            location_history = list(self.location_history2)[-3:]
        if len(detection_history) < 2:
            return False

        # Check if detected in 2 consecutive frames
        if detection_history[-1] and detection_history[-2]:
            return True

        # Check if detected in 2 of the last 3 frames at a similar location
        if sum(detection_history) >= 2:
            # Further check for spatial consistency
            locations = [loc for loc in location_history if loc is not None]
            if len(locations) >= 2:
                # Example condition: check if locations are within some distance threshold
                dist_threshold = MAX_PUCK_DEV  # Define a suitable threshold for your application
                last_location = locations[-1]
                for loc in locations[:-1]:
                    if distance(last_location, loc) <= dist_threshold:
                        return True
        return False

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'

        """
        self.team, self.num_players = team, num_players

        return ["tux"] * num_players

    def reset(self):
        self.last_puck_screen_center1 = torch.tensor([0, 0])
        self.last_puck_screen_center2 = torch.tensor([0, 0])
        self.kickoff_steps_left = KICKOFF_EVENT_DURATION
        self.working_normaly1 = True
        self.working_normaly2 = True

        self.detection_history1 = deque(maxlen=4)
        self.location_history1 = deque(maxlen=4)
        self.last_positive_location1 = torch.tensor([0, 0])
        self.search_counter1 = 0
        self.reverse_timer1 = 0

        self.detection_history2 = deque(maxlen=4)
        self.location_history2 = deque(maxlen=4)
        self.last_positive_location2 = torch.tensor([0, 0])
        self.search_counter2 = 0
        self.reverse_timer2 = 0
        self.search_rear1 = 0
        self.search_rear2 = 0
        self.velocity_history1 = deque(maxlen=12)
        self.velocity_history2 = deque(maxlen=12)
        self.flag_prob_history1 = deque(maxlen=7)
        self.flag_prob_history2 = deque(maxlen=7)
        self.use_puck1 = True
        self.use_puck2 = True
        self.reversal_aim1 = torch.tensor([0, 1])
        self.reversal_aim2 = torch.tensor([0, 1])

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # try:
        velocity1, player_id1, max_steer_angle1, kart_center1, kart_front1, projection1, view1 = get_player_info(0, player_state)
        velocity2, player_id2, max_steer_angle2, kart_center2, kart_front2, projection2, view2 = get_player_info(1, player_state)

        ##  calculate derived infos from state
        kart_direction1 = (kart_front1 - kart_center1) / torch.norm(kart_front1 - kart_center1)
        kart_direction2 = (kart_front2 - kart_center2) / torch.norm(kart_front2 - kart_center2)
        # kart_angle1 = torch.atan2(kart_direction1[1], kart_direction1[0])  # angle from x-axis in radians (?)
        # kart_angle2 = torch.atan2(kart_direction2[1], kart_direction2[0])  # from -pi to pi
        ## get static infos
        goal_center = get_goal(player_id_to_team_id(player_id1))
        own_goal_center = get_own_goal(player_id_to_team_id(player_id1))
        ## caclulate derived infos from static infos
        dist_goal1 = distance(goal_center, kart_center1)
        dist_own_goal1 = distance(own_goal_center, kart_center1)
        dist_goal2 = distance(goal_center, kart_center2)
        dist_own_goal2 = distance(own_goal_center, kart_center2)
        goal_direction1 = (goal_center - kart_center1) / torch.norm(goal_center - kart_center1)
        goal_direction2 = (goal_center - kart_center2) / torch.norm(goal_center - kart_center2)
        own_goal_direction1 = (own_goal_center - kart_center1) / torch.norm(own_goal_center - kart_center1)
        own_goal_direction2 = (own_goal_center - kart_center2) / torch.norm(own_goal_center - kart_center2)
        angle_kart_goal1 = torch.acos(torch.dot(kart_direction1, goal_direction1))
        angle_kart_goal2 = torch.acos(torch.dot(kart_direction2, goal_direction2))
        angle_kart_own_goal1 = torch.acos(torch.dot(kart_direction1, own_goal_direction1))
        angle_kart_own_goal2 = torch.acos(torch.dot(kart_direction2, own_goal_direction2))

        center_direction1 = (torch.tensor([0, 0]) - kart_center1) / torch.norm(torch.tensor([0, 0]) - kart_center1)
        center_direction2 = (torch.tensor([0, 0]) - kart_center2) / torch.norm(torch.tensor([0, 0]) - kart_center2)

        angle_kart_center1 = torch.acos(torch.dot(kart_direction1, center_direction1))
        angle_kart_center2 = torch.acos(torch.dot(kart_direction2, center_direction2))
        angle_kart_center1_left_or_right = -torch.sign(torch_2d_cross(kart_direction1, center_direction1))
        angle_kart_center2_left_or_right = -torch.sign(torch_2d_cross(kart_direction2, center_direction2))

        angle_kart_goal1_left_or_right = -torch.sign(torch_2d_cross(kart_direction1, goal_direction1))
        angle_kart_goal1_aim_offset = torch.rad2deg(angle_kart_goal1_left_or_right * angle_kart_goal1)
        angle_kart_goal2_left_or_right = -torch.sign(torch_2d_cross(kart_direction2, goal_direction2))
        angle_kart_goal2_aim_offset = torch.rad2deg(angle_kart_goal2_left_or_right * angle_kart_goal2)

        angle_kart_own_goal1_left_or_right = -torch.sign(torch_2d_cross(kart_direction1, own_goal_direction1))
        angle_kart_own_goal1_aim_offset = torch.rad2deg(angle_kart_own_goal1_left_or_right * angle_kart_own_goal1)
        angle_kart_own_goal2_left_or_right = -torch.sign(torch_2d_cross(kart_direction2, own_goal_direction2))
        angle_kart_own_goal2_aim_offset = torch.rad2deg(angle_kart_own_goal2_left_or_right * angle_kart_own_goal2)

        # predict
        images = [self.transform(Image.fromarray(i))[None, :, :, :] for i in player_image]
        images = torch.cat(images).to(device)  # resize imag
        with torch.no_grad():
            centers, depths, flags = self.model(images)
        centers, depths, flags = centers.to("cpu"), depths.to("cpu"), flags.to("cpu")

        # has anyone seen the puck?
        flag1, flag2 = torch.sigmoid(flags).squeeze() > FLAG_THR  # adapt FLAG_THR=0.5 to something else?
        # Update history
        self.flag_prob_history1.append(flag1)
        self.flag_prob_history2.append(flag2)
        self.location_fifo1.add(kart_center1)
        self.location_fifo2.add(kart_center2)
        self.velocity_history1.append(velocity1.item())
        self.velocity_history2.append(velocity2.item())
        self.detection_history1.append(flag1)
        self.location_history1.append(centers[0] if flag1 else None)
        self.detection_history2.append(flag2)
        self.location_history2.append(centers[1] if flag1 else None)

        # Check if all except at most one tensor are above the threshold
        # Determine if it's a valid detection          # if deviation from last location is too great, ignore it via:
        ##flag1 = self.confirm_detection(player_id=1)
        # flag2 = self.confirm_detection(player_id=2)

        # if flag1 and self.last_puck_screen_center1 != None and torch.abs(centers[0] - self.last_puck_screen_center1) > MAX_PUCK_DEV:
        #     centers[0] = self.last_puck_screen_center1
        #     self.use_puck1 = False
        #     flag1 = False
        # else:
        #     self.use_puck1 = True
        #     self.last_puck_screen_center1 == None

        # if flag2 and self.last_puck_screen_center1 != None and torch.abs(centers[0] - self.last_puck_screen_center2) > MAX_PUCK_DEV:
        #     centers[1] = self.last_puck_screen_center2
        #     self.use_puck2 = False
        #     flag2 = False
        # else:
        #     self.use_puck2 = True
        #     self.last_puck_screen_center2 == None

        # Determine if it's a valid detection          # if deviation from last location is too great, ignore it via:
        flag1 = self.confirm_detection(player_id=1)
        flag2 = self.confirm_detection(player_id=2)
        # Update last positive location if confirmed
        if flag1:
            self.last_positive_location1 = centers[0]
        else:
            centers[0] = self.last_positive_location1
        if flag2:
            self.last_positive_location2 = centers[1]
        else:
            centers[1] = self.last_positive_location2

        ## REMOVED: puck tracking
        # if flag1:
        #     if self.working_normaly1 and distance(self.last_puck_screen_center1, centers[0]) > MAX_PUCK_DEV:
        #         centers[0] = self.last_puck_screen_center1
        #         self.working_normaly1 = False
        #     else:
        #         self.working_normaly1 = True
        # self.last_puck_screen_center1 = centers[0]

        # if flag2:
        #     if self.working_normaly2 and distance(self.last_puck_screen_center2, centers[1]) > MAX_PUCK_DEV:
        #         centers[1] = self.last_puck_screen_center2
        #         self.working_normaly2 = False
        #     else:
        #         self.working_normaly2 = True
        # self.last_puck_screen_center2 = centers[1]

        # centers[1] = self.last_puck_screen_center2

        # REMOVED: screen to world
        # wx1, wy1, wz1 = screen_to_world(centers[0][1], centers[0][0], depths[0], view1, projection1)
        # wx2, wy2, wz2 = screen_to_world(centers[1][1], centers[1][0], depths[1], view2, projection2)
        # wy is the unimportant value = discard and rename
        # wx1, wy1, wz1 = wx1, wz1, wy1
        # wx2, wy2, wz2 = wx2, wz2, wy2

        aim_point1 = centers[0]
        aim_point2 = centers[1]

        ## REMOVED: try to calculate puck position
        # distance_to_puck1 = distance(kart_center1, torch.tensor([wx1, wy1]))
        # distance_to_puck2 = distance(kart_center2, torch.tensor([wx2, wy2]))
        # puck_center1 = torch.tensor([wy1, wx1])
        # puck_center2 = torch.tensor([wy2, wx2])

        ## REMOVED: try to calculate perfect aim point based on world coordinates
        # puck_direction1 = (puck_center1 - kart_center1) / torch.norm(puck_center1 - kart_center1)
        # puck_direction2 = (puck_center2 - kart_center2) / torch.norm(puck_center2 - kart_center2)
        # angle_kart_puck1 = torch.acos(torch.dot(kart_direction1, puck_direction1))
        # angle_kart_puck2 = torch.acos(torch.dot(kart_direction2, puck_direction2))
        # perfect_hit_point1, offset1 = calculate_hit_point(kart_center1, puck_center1, goal_center, get_size_of_puck())
        # perfect_hit_point2, offset1 = calculate_hit_point(kart_center2, puck_center2, goal_center, get_size_of_puck())
        # perfect_hit_point1_4d = torch.tensor([perfect_hit_point1[0], 0.37, perfect_hit_point1[1], 1])
        # perfect_hit_point2_4d = torch.tensor([perfect_hit_point2[0], 0.37, perfect_hit_point2[1], 1])
        # aim_point1_from_world = world_to_screen(perfect_hit_point1_4d, projection1, view1)
        # aim_point2_from_world = world_to_screen(perfect_hit_point2_4d, projection2, view2)

        nitro1, nitro2 = False, False
        reverse1, reverse2 = False, False
        brake1, brake2 = False, False
        player1_state, player2_state = "KICKOFF", "KICKOFF"

        # PER PLAYER ACTION - KART 1
        # HANDLE SPECIAL CASES:
        ### Kickoff
        if (
            in_start_position(kart_center1)
            and in_start_position(kart_center2)
            and all(vel < 0.2 for vel in self.velocity_history1)
            and all(vel < 0.2 for vel in self.velocity_history2)
            and in_goal(kart_center1) == False
            and in_goal(kart_center2) == False
        ) or self.kickoff_steps_left > 0:
            acceleration1 = 1.0
            nitro1 = True
            brake1 = False
            player1_state = "KICKOFF"
            aim1 = aim_point1
            if self.kickoff_steps_left == 0:
                self.reset()
            elif velocity1 > 0.2 and velocity2 > 0.2:
                self.kickoff_steps_left -= 1
        ## in goal or stuck
        elif in_goal(kart_center1) == True or (self.location_fifo1.stuck(0.1) or self.reversal_counter1 > 0):
            player1_state = "IN GOAL" if in_goal(kart_center1) else "STUCK"
            if torch.abs(angle_kart_center1) < torch.deg2rad(torch.tensor(25)):  # within almost +-40°
                self.reversal_counter1 = 0
                reverse1 = False
                acceleration1 = 0.5
                was_puck_detected1 = self.confirm_detection(player_id=1)
                aim1 = aim_point1 if was_puck_detected1 else torch.tensor([0, 0])
            elif self.reversal_counter1 > UNSTUCK_EVENT_DURATION / 2:
                reverse1 = True
                acceleration1 = 0.0
                self.reversal_counter1 -= 1
                # aim1  = angle_kart_center1_left_or_right

                # print("1", aim1)
            elif self.reversal_counter1 > 0:
                reverse1 = False
                acceleration1 = 1.0
                self.reversal_counter1 -= 1
                # aim1 = angle_kart_center1_left_or_right
            else:
                reverse1 = True
                acceleration1 = 0.0
                self.reversal_counter1 -= 1
                # aim1 = angle_kart_center1_left_or_right
                self.reversal_counter1 = UNSTUCK_EVENT_DURATION

        ### STANDARD CASE - puck in screen
        elif flag1:
            player1_state = "STANDARD"
            brake1 = False
            # if dist_own_goal1 < 10 or torch.abs(angle_kart_own_goal1) < torch.deg2rad(torch.tensor(15)):
            #     # do the opposite of goal shooting
            #     distance_weighting1 = distance_weighting_function(dist_own_goal1)
            #     distance_weighting1 = torch.clip(distance_weighting1, 1 / 40, 1 / 40)
            #     aim_point1 = aim_point1 - distance_weighting1 * torch.sign(aim_point1 - angle_kart_own_goal1_aim_offset / 100)

            if 22 < torch.abs(angle_kart_goal1_aim_offset) < 122:
                distance_weighting1 = distance_weighting_function(dist_goal1)
                aim_point1 = aim_point1 + distance_weighting1 * torch.sign(aim_point1 - angle_kart_goal1_aim_offset / 100)
            else:
                aim_point1 = aim_point1
            certaintly_a_puck1 = sum(1 for x in self.flag_prob_history1 if x <= 0.7) <= 1
            acceleration1 = 0.75 if norm(torch.tensor(player_state[0]["kart"]["velocity"])) < TARGET_VELOCITY else 0

        ### PUCK VANISHED FOR LESS THAN 5 FRAMES
        elif len(self.detection_history1) >= 2 and all(islice(self.detection_history1, len(self.detection_history1) - 2)):
            player1_state = "KEEP GOING"
            self.keep_tracking1 = 5
            brake1 = False
            acceleration1 = 0.5
            aim_point1 = aim_point1
        elif self.keep_tracking1 > 0:
            player1_state = "KEEP GOING"
            self.keep_tracking1 -= 1
            brake1 = False
            acceleration1 = 0.5
            aim_point1 = aim_point1
        ### puck not in screen
        else:
            player1_state = "PUCK NOT VISIBLE1"
            if self.search_counter1 > 0:
                player1_state = "PUCK NOT VISIBLE2"
                self.search_counter1 -= 1
                aim_point1 = torch.tensor([0, 1]) * angle_kart_goal1_aim_offset / 100
                acceleration1 = 0.2
                brake1 = False
                # self.search_rear1 -= 1

            elif dist_own_goal1 > 40:  # changed
                aim_point1 = torch.tensor([0, 1]) * angle_kart_own_goal1_aim_offset / 100
                acceleration1 = 0.4  # changed0.2
                brake1 = False  # changed
                # self.search_rear1 = 0
            else:
                self.search_counter1 = SEARCH_STEPS
                aim_point1 = torch.tensor([0, 1]) * angle_kart_own_goal1_aim_offset / 100
                acceleration1 = 0.4  # changed
                brake1 = False

        if reverse1:
            acceleration1 = 0.0
            brake1 = True
        aim1 = aim_point1

        # PER PLAYER ACTION - KART 2
        # HANDLE SPECIAL CASES:
        ### Kickoff
        if self.kickoff_steps_left > 0:
            acceleration2 = 1.0
            nitro2 = True
            brake2 = False
            aim2 = aim_point2
            player2_state = "KICKOFF"  # timer is handeled by kart1
        ## in goal or stuck
        elif in_goal(kart_center2) == True or self.location_fifo2.stuck(0.1) or self.reversal_counter2 > 0:
            player2_state = "IN GOAL" if in_goal(kart_center2) else "STUCK"
            if torch.abs(angle_kart_center2) < torch.deg2rad(torch.tensor(25)):  # within almost +-
                self.reversal_counter2 -= 1
                reverse2 = False
                acceleration2 = 0.5
                was_puck_detected2 = self.confirm_detection(player_id=2)
                aim2 = aim_point2 if was_puck_detected2 else torch.tensor([0, 0])
            elif self.reversal_counter2 > UNSTUCK_EVENT_DURATION / 2:
                reverse2 = True
                acceleration2 = 0.0
                self.reversal_counter2 -= 1
                # aim2 = angle_kart_center2_left_or_right
                # print("2", aim2)
            elif self.reversal_counter2 > 0:
                reverse2 = False
                acceleration2 = 1.0
                self.reversal_counter2 -= 1
                # aim2 = angle_kart_center2_left_or_right
            else:
                reverse2 = True
                acceleration2 = 0.0
                self.reversal_counter2 -= 1
                # aim2 = angle_kart_center2_left_or_right
                self.reversal_counter2 = UNSTUCK_EVENT_DURATION
        ### STANDARD CASE - puck in screen
        elif flag2:
            player2_state = "STANDARD"
            # if dist_own_goal2 < 15 or torch.abs(angle_kart_own_goal2) < torch.deg2rad(torch.tensor(15)):  # within almost +-40°
            #     # do the opposite of goal shooting
            #     distance_weighting2 = distance_weighting_function(dist_own_goal1)
            #     distance_weighting2 = torch.clip(distance_weighting2, 1 / 40, 1 / 40)
            #     aim_point2 = aim_point2 - distance_weighting2 * torch.sign(aim_point2 - angle_kart_own_goal2_aim_offset / 100)
            if 22 < torch.abs(angle_kart_goal2_aim_offset) < 122:
                distance_weighting2 = distance_weighting_function(dist_goal2)
                aim_point2 = aim_point2 + distance_weighting2 * torch.sign(aim_point2 - angle_kart_goal2_aim_offset / 100)
            else:
                aim_point2 = aim_point2
            brake2 = False
            certaintly_a_puck2 = sum(1 for x in self.flag_prob_history2 if x <= 0.7) <= 1
            acceleration2 = 0.75 if norm(torch.tensor(player_state[1]["kart"]["velocity"])) < TARGET_VELOCITY else 0
        ### PUCK VANISHED FOR LESS THAN 9 FRAMES
        elif len(self.detection_history2) >= 2 and all(islice(self.detection_history2, len(self.detection_history2) - 2)):
            player2_state = "KEEP GOING"
            self.keep_tracking2 = 5
            brake2 = False
            acceleration2 = 0.5
            aim_point2 = aim_point2
        elif self.keep_tracking2 > 0:
            player2_state = "KEEP GOING"
            self.keep_tracking2 -= 1
            brake2 = False
            acceleration2 = 0.5
            aim_point2 = aim_point2
        ### puck not in screen
        else:
            player2_state = "PUCK NOT VISIBLE1"
            if self.search_counter2 > 0:
                player2_state = "PUCK NOT VISIBLE2"
                self.search_counter2 -= 1
                aim_point2 = torch.tensor([0, 1]) * angle_kart_goal2_aim_offset / 100
                acceleration2 = 0.2
                brake2 = False
                ##self.search_rear2 -= 1

            elif dist_own_goal2 > 40:  # changed
                aim_point2 = torch.tensor([0, 1]) * angle_kart_own_goal2_aim_offset / 100
                acceleration2 = 0.4  # changed
                brake2 = False  # changed
                # self.search_rear2 = 0

            else:
                self.search_counter2 = SEARCH_STEPS
                aim_point2 = torch.tensor([0, 1]) * angle_kart_own_goal2_aim_offset / 100
                acceleration2 = 0.4  # changed
                brake2 = True  # changed
                # self.search_rear2 -= 1

        if reverse2:
            acceleration2 = 0.0
            brake2 = True
        aim2 = aim_point2

        # print("!", player1_state, ": ", aim1, "    ", player2_state, ": ", aim2, "   ", self.detection_history1, self.detection_history2)

        steer1 = float(np.clip(aim1[1].item() * STEER_MULTIPLIER, -1, 1))
        steer2 = float(np.clip(aim2[1].item() * STEER_MULTIPLIER, -1, 1))
        drift1 = False if flag1 else bool(np.abs(steer1) > DRIFT_THR)
        drift2 = False if flag2 else bool(np.abs(steer2) > DRIFT_THR)

        self.step += 1
        self.reversal_counter1 -= 1
        self.reversal_counter2 -= 1
        action1_dict = {"steer": steer1, "acceleration": acceleration1, "brake": brake1, "drift": drift1, "nitro": nitro1, "rescue": False, "fire": False}
        action2_dict = {"steer": steer2, "acceleration": acceleration2, "brake": brake2, "drift": drift2, "nitro": nitro2, "rescue": False, "fire": False}

        if DEBUG_AND_OVERLAY:
            player1_drift_str = "+" if drift1 else ""
            player2_drift_str = "+" if drift2 else ""
            kart1_center_str = f"({tuple(kart_center1.numpy())[0]:.2f}, {tuple(kart_center1.numpy())[1]:.2f})"
            kart2_center_str = f"({tuple(kart_center2.numpy())[0]:.2f}, {tuple(kart_center2.numpy())[1]:.2f})"
            overlay_data["player1_circle"] = aim1
            overlay_data["player2_circle"] = aim2
            overlay_data["player1_flag"] = flag1
            overlay_data["player2_flag"] = flag2
            overlay_data["player1_steer"] = "Steer: " + f"{steer1:.2f}" + player1_drift_str
            overlay_data["player2_steer"] = "Steer: " + f"{steer2:.2f}" + player2_drift_str
            overlay_data["player1_state"] = player1_state
            overlay_data["player2_state"] = player2_state
            overlay_data["player1_loc"] = kart1_center_str
            overlay_data["player2_loc"] = kart2_center_str
            overlay_data["player1_depth"] = "Depth: " + f"{float(depths[0].numpy()):.4f}"
            overlay_data["player2_depth"] = "Depth: " + f"{float(depths[1].numpy()):.4f}"
            overlay_data["player1_p_est"] = f"{float(torch.sigmoid(flags[0]).numpy()):.3f}"
            overlay_data["player2_p_est"] = f"{float(torch.sigmoid(flags[1]).numpy()):.3f}"
        return [action1_dict, action2_dict]

        # except Exception as e:
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(exc_type, fname, exc_tb.tb_lineno)
        #     print(e)
