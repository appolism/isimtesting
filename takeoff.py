import math
import numpy as np

import omni.physx
import omni.usd

from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


# ============================================================
# USER CONFIGURATION
# ============================================================

# Must be the exact prim containing Rigid Body API.
DRONE_BODY_PATH = "/World/cf2x/body"

# Used if the rigid body does not have an authored mass.
DRONE_MASS_KG = 0.027

TARGET_HEIGHT_M = 1.0
GRAVITY_M_S2 = 9.81


# ============================================================
# TAKEOFF PROFILE
# ============================================================

# The commanded height rises gradually instead of instantly
# jumping from the starting height to TARGET_HEIGHT_M.
TAKEOFF_SPEED_M_S = 0.35

# Wait briefly before beginning the takeoff ramp.
START_DELAY_S = 0.15


# ============================================================
# POSITION CONTROL GAINS
# ============================================================

# Vertical controller
KP_Z = 3.0
KD_Z = 3.5

# Horizontal position hold
KP_XY = 1.0
KD_XY = 2.2

# Limits
MAX_HORIZONTAL_ACCEL_M_S2 = 1.5
MAX_UPWARD_CONTROL_ACCEL_M_S2 = 4.0
MAX_DOWNWARD_CONTROL_ACCEL_M_S2 = 4.0

MAX_TOTAL_FORCE_MULTIPLIER = 1.7


# ============================================================
# ATTITUDE CONTROL GAINS
# ============================================================

# Rotation-matrix geometric controller.
#
# The controller holds the exact orientation the drone had when
# simulation began.
ATTITUDE_KP = np.array(
    [0.0025, 0.0025, 0.0012],
    dtype=np.float64,
)

ATTITUDE_KD = np.array(
    [0.0009, 0.0009, 0.0005],
    dtype=np.float64,
)

MAX_TORQUE_XY_NM = 0.0012
MAX_TORQUE_Z_NM = 0.0006


# ============================================================
# DAMPING AND FILTERING
# ============================================================

LINEAR_DAMPING = 0.10
ANGULAR_DAMPING = 0.15

# Low-pass filtering reduces noise from pose differentiation.
LINEAR_VELOCITY_FILTER = 0.25
ANGULAR_VELOCITY_FILTER = 0.20

# Position and attitude dead zones prevent constant tiny
# corrections around the target.
XY_DEAD_ZONE_M = 0.005
Z_DEAD_ZONE_M = 0.005

ATTITUDE_DEAD_ZONE_RAD = math.radians(0.15)
ANGULAR_RATE_DEAD_ZONE_RAD_S = math.radians(0.5)

PRINT_INTERVAL_S = 0.5


# ============================================================
# CLEAN UP AN EARLIER COPY
# ============================================================

try:
    cf2x_takeoff_subscription = None
except NameError:
    pass

try:
    cf2x_force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    cf2x_torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
except Exception:
    pass


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(float(value), maximum))


def clamp_vector_magnitude(vector, maximum_magnitude):
    vector = np.asarray(vector, dtype=np.float64)
    magnitude = np.linalg.norm(vector)

    if magnitude > maximum_magnitude and magnitude > 1e-12:
        return vector * (maximum_magnitude / magnitude)

    return vector


def apply_dead_zone(value, dead_zone):
    if abs(float(value)) < dead_zone:
        return 0.0

    return float(value)


def quaternion_normalize(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    magnitude = np.linalg.norm(quaternion)

    if magnitude < 1e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    return quaternion / magnitude


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion [w, x, y, z] into a body-to-world
    rotation matrix.
    """

    w, x, y, z = quaternion_normalize(quaternion)

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rotation_vector_from_matrix(rotation_matrix):
    """
    Convert a rotation matrix into an axis-angle rotation vector.
    The returned vector is expressed in the matrix's coordinate
    frame.
    """

    matrix = np.asarray(rotation_matrix, dtype=np.float64)

    trace_value = np.trace(matrix)

    cosine_angle = clamp(
        (trace_value - 1.0) * 0.5,
        -1.0,
        1.0,
    )

    angle = math.acos(cosine_angle)

    if angle < 1e-7:
        return np.array(
            [
                0.5 * (matrix[2, 1] - matrix[1, 2]),
                0.5 * (matrix[0, 2] - matrix[2, 0]),
                0.5 * (matrix[1, 0] - matrix[0, 1]),
            ],
            dtype=np.float64,
        )

    sine_angle = math.sin(angle)

    if abs(sine_angle) < 1e-7:
        # Near 180 degrees. Use the diagonal to obtain a stable
        # approximate rotation axis.
        axis = np.sqrt(
            np.maximum(
                np.diag(matrix) + 1.0,
                0.0,
            )
            * 0.5
        )

        if np.linalg.norm(axis) < 1e-8:
            return np.zeros(3, dtype=np.float64)

        axis = axis / np.linalg.norm(axis)

        return axis * angle

    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )

    axis = axis / (2.0 * sine_angle)

    return axis * angle


# ============================================================
# POSE FUNCTIONS
# ============================================================

def get_world_pose():
    """
    Returns:
        position_world: [x, y, z]
        orientation: quaternion [w, x, y, z]
        rotation_body_to_world: 3x3 matrix
    """

    xform_cache = UsdGeom.XformCache()
    world_transform = xform_cache.GetLocalToWorldTransform(
        drone_prim
    )

    translation = world_transform.ExtractTranslation()
    rotation = world_transform.ExtractRotation()
    gf_quaternion = rotation.GetQuat()
    imaginary = gf_quaternion.GetImaginary()

    position_world = np.array(
        [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ],
        dtype=np.float64,
    )

    orientation = quaternion_normalize(
        np.array(
            [
                float(gf_quaternion.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ],
            dtype=np.float64,
        )
    )

    rotation_body_to_world = quaternion_to_rotation_matrix(
        orientation
    )

    return (
        position_world,
        orientation,
        rotation_body_to_world,
    )


def calculate_body_angular_velocity(
    previous_rotation_body_to_world,
    current_rotation_body_to_world,
    dt,
):
    """
    Estimate angular velocity in the current drone body frame.

    Relative rotation:
        R_delta = R_previous^T * R_current
    """

    if previous_rotation_body_to_world is None or dt <= 0.0:
        return np.zeros(3, dtype=np.float64)

    relative_rotation_body = (
        previous_rotation_body_to_world.T
        @ current_rotation_body_to_world
    )

    rotation_vector_body = rotation_vector_from_matrix(
        relative_rotation_body
    )

    angular_velocity_body = rotation_vector_body / dt

    if (
        not np.all(np.isfinite(angular_velocity_body))
        or np.linalg.norm(angular_velocity_body) > 100.0
    ):
        return np.zeros(3, dtype=np.float64)

    return angular_velocity_body


# ============================================================
# STAGE AND RIGID BODY
# ============================================================

stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError("No USD stage is currently open.")

drone_prim = stage.GetPrimAtPath(DRONE_BODY_PATH)

if not drone_prim.IsValid():
    raise RuntimeError(
        f"Could not find a prim at {DRONE_BODY_PATH}. "
        "Set DRONE_BODY_PATH to the actual drone rigid body."
    )

rigid_body_api = UsdPhysics.RigidBodyAPI(drone_prim)

if not rigid_body_api:
    raise RuntimeError(
        f"{DRONE_BODY_PATH} does not contain Rigid Body API. "
        "Use the exact prim containing the rigid body."
    )

rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
rigid_body_api.CreateKinematicEnabledAttr().Set(False)


# ============================================================
# MASS
# ============================================================

controller_mass_kg = DRONE_MASS_KG

mass_api = UsdPhysics.MassAPI(drone_prim)

if mass_api:
    mass_attr = mass_api.GetMassAttr()

    if mass_attr:
        authored_mass = mass_attr.Get()

        if authored_mass is not None:
            authored_mass = float(authored_mass)

            if authored_mass > 0.0:
                controller_mass_kg = authored_mass

if controller_mass_kg <= 0.0:
    raise RuntimeError(
        "The drone mass must be greater than zero."
    )


# ============================================================
# PHYSX RIGID-BODY SETTINGS
# ============================================================

physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(
    drone_prim
)

if not physx_rigid_body_api:
    physx_rigid_body_api = (
        PhysxSchema.PhysxRigidBodyAPI.Apply(drone_prim)
    )

physx_rigid_body_api.CreateLinearDampingAttr().Set(
    LINEAR_DAMPING
)

physx_rigid_body_api.CreateAngularDampingAttr().Set(
    ANGULAR_DAMPING
)

# Keep every rotational axis unlocked.
for attribute_name in (
    "physxRigidBody:lockedRotAxisX",
    "physxRigidBody:lockedRotAxisY",
    "physxRigidBody:lockedRotAxisZ",
):
    attribute = drone_prim.GetAttribute(attribute_name)

    if attribute and attribute.IsValid():
        attribute.Set(False)


# ============================================================
# FORCE AND TORQUE API
# ============================================================

force_api = PhysxSchema.PhysxForceAPI(drone_prim)

if not force_api:
    force_api = PhysxSchema.PhysxForceAPI.Apply(drone_prim)

cf2x_force_attr = force_api.CreateForceAttr()
cf2x_torque_attr = force_api.CreateTorqueAttr()

force_api.CreateForceEnabledAttr().Set(True)

# Both vectors are world-frame values.
force_api.CreateWorldFrameEnabledAttr().Set(True)

# Use force and torque units rather than acceleration units.
force_api.CreateModeAttr().Set("force")

cf2x_force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
cf2x_torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


# ============================================================
# CONTROLLER STATE
# ============================================================

initial_position_world = None
target_position_world = None

target_rotation_body_to_world = None

previous_position_world = None
previous_rotation_body_to_world = None

filtered_linear_velocity_world = np.zeros(
    3,
    dtype=np.float64,
)

filtered_angular_velocity_body = np.zeros(
    3,
    dtype=np.float64,
)

elapsed_time_s = 0.0
print_timer_s = 0.0


# ============================================================
# PHYSICS CALLBACK
# ============================================================

def cf2x_physics_step(dt):
    global initial_position_world
    global target_position_world
    global target_rotation_body_to_world
    global previous_position_world
    global previous_rotation_body_to_world
    global filtered_linear_velocity_world
    global filtered_angular_velocity_body
    global elapsed_time_s
    global print_timer_s

    if dt is None or dt <= 0.0:
        return

    dt = float(dt)

    elapsed_time_s += dt
    print_timer_s += dt

    (
        position_world,
        orientation,
        rotation_body_to_world,
    ) = get_world_pose()

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    if initial_position_world is None:
        initial_position_world = position_world.copy()

        target_position_world = np.array(
            [
                initial_position_world[0],
                initial_position_world[1],
                initial_position_world[2],
            ],
            dtype=np.float64,
        )

        # Hold the exact starting orientation. This does not
        # assume that the drone's local axes match world axes.
        target_rotation_body_to_world = (
            rotation_body_to_world.copy()
        )

        previous_position_world = position_world.copy()
        previous_rotation_body_to_world = (
            rotation_body_to_world.copy()
        )

        print("")
        print("CF2X controller initialized.")
        print(
            "Initial position: "
            f"({initial_position_world[0]:.3f}, "
            f"{initial_position_world[1]:.3f}, "
            f"{initial_position_world[2]:.3f})"
        )
        print(f"Final height: {TARGET_HEIGHT_M:.3f} m")
        print("Rotation remains fully unlocked.")

        return

    # --------------------------------------------------------
    # VELOCITY ESTIMATION AND FILTERING
    # --------------------------------------------------------

    measured_linear_velocity_world = (
        position_world - previous_position_world
    ) / dt

    if (
        not np.all(np.isfinite(measured_linear_velocity_world))
        or np.linalg.norm(measured_linear_velocity_world) > 20.0
    ):
        measured_linear_velocity_world = np.zeros(
            3,
            dtype=np.float64,
        )

    measured_angular_velocity_body = (
        calculate_body_angular_velocity(
            previous_rotation_body_to_world,
            rotation_body_to_world,
            dt,
        )
    )

    filtered_linear_velocity_world = (
        LINEAR_VELOCITY_FILTER
        * measured_linear_velocity_world
        + (1.0 - LINEAR_VELOCITY_FILTER)
        * filtered_linear_velocity_world
    )

    filtered_angular_velocity_body = (
        ANGULAR_VELOCITY_FILTER
        * measured_angular_velocity_body
        + (1.0 - ANGULAR_VELOCITY_FILTER)
        * filtered_angular_velocity_body
    )

    previous_position_world = position_world.copy()
    previous_rotation_body_to_world = (
        rotation_body_to_world.copy()
    )

    # --------------------------------------------------------
    # SMOOTH TAKEOFF TARGET
    # --------------------------------------------------------

    if elapsed_time_s <= START_DELAY_S:
        commanded_height = initial_position_world[2]
    else:
        takeoff_elapsed = elapsed_time_s - START_DELAY_S

        commanded_height = (
            initial_position_world[2]
            + TAKEOFF_SPEED_M_S * takeoff_elapsed
        )

        commanded_height = min(
            commanded_height,
            TARGET_HEIGHT_M,
        )

    target_position_world[0] = initial_position_world[0]
    target_position_world[1] = initial_position_world[1]
    target_position_world[2] = commanded_height

    # --------------------------------------------------------
    # POSITION ERROR
    # --------------------------------------------------------

    position_error_world = (
        target_position_world - position_world
    )

    position_error_world[0] = apply_dead_zone(
        position_error_world[0],
        XY_DEAD_ZONE_M,
    )

    position_error_world[1] = apply_dead_zone(
        position_error_world[1],
        XY_DEAD_ZONE_M,
    )

    position_error_world[2] = apply_dead_zone(
        position_error_world[2],
        Z_DEAD_ZONE_M,
    )

    # --------------------------------------------------------
    # HORIZONTAL POSITION HOLD
    # --------------------------------------------------------

    horizontal_acceleration_command = (
        KP_XY * position_error_world[0:2]
        - KD_XY * filtered_linear_velocity_world[0:2]
    )

    horizontal_acceleration_command = (
        clamp_vector_magnitude(
            horizontal_acceleration_command,
            MAX_HORIZONTAL_ACCEL_M_S2,
        )
    )

    # --------------------------------------------------------
    # VERTICAL CONTROL
    # --------------------------------------------------------

    vertical_control_acceleration = (
        KP_Z * position_error_world[2]
        - KD_Z * filtered_linear_velocity_world[2]
    )

    vertical_control_acceleration = clamp(
        vertical_control_acceleration,
        -MAX_DOWNWARD_CONTROL_ACCEL_M_S2,
        MAX_UPWARD_CONTROL_ACCEL_M_S2,
    )

    desired_acceleration_world = np.array(
        [
            horizontal_acceleration_command[0],
            horizontal_acceleration_command[1],
            GRAVITY_M_S2 + vertical_control_acceleration,
        ],
        dtype=np.float64,
    )

    desired_acceleration_world[2] = max(
        0.0,
        desired_acceleration_world[2],
    )

    # --------------------------------------------------------
    # WORLD-FRAME FORCE
    # --------------------------------------------------------

    desired_force_world = (
        controller_mass_kg * desired_acceleration_world
    )

    hover_force_n = controller_mass_kg * GRAVITY_M_S2

    desired_force_world = clamp_vector_magnitude(
        desired_force_world,
        hover_force_n * MAX_TOTAL_FORCE_MULTIPLIER,
    )

    # --------------------------------------------------------
    # GEOMETRIC ATTITUDE ERROR
    # --------------------------------------------------------
    #
    # Current rotation:
    #     R
    #
    # Desired rotation:
    #     R_d
    #
    # Body-frame geometric error:
    #
    # e_R = 0.5 * vee(R_d^T R - R^T R_d)
    #
    # Corrective torque:
    #
    # tau_body = -Kp * e_R - Kd * omega_body
    # --------------------------------------------------------

    attitude_error_matrix = 0.5 * (
        target_rotation_body_to_world.T
        @ rotation_body_to_world
        - rotation_body_to_world.T
        @ target_rotation_body_to_world
    )

    attitude_error_body = np.array(
        [
            attitude_error_matrix[2, 1],
            attitude_error_matrix[0, 2],
            attitude_error_matrix[1, 0],
        ],
        dtype=np.float64,
    )

    for axis in range(3):
        attitude_error_body[axis] = apply_dead_zone(
            attitude_error_body[axis],
            ATTITUDE_DEAD_ZONE_RAD,
        )

        filtered_angular_velocity_body[axis] = apply_dead_zone(
            filtered_angular_velocity_body[axis],
            ANGULAR_RATE_DEAD_ZONE_RAD_S,
        )

    desired_torque_body = (
        -ATTITUDE_KP * attitude_error_body
        - ATTITUDE_KD * filtered_angular_velocity_body
    )

    desired_torque_body[0] = clamp(
        desired_torque_body[0],
        -MAX_TORQUE_XY_NM,
        MAX_TORQUE_XY_NM,
    )

    desired_torque_body[1] = clamp(
        desired_torque_body[1],
        -MAX_TORQUE_XY_NM,
        MAX_TORQUE_XY_NM,
    )

    desired_torque_body[2] = clamp(
        desired_torque_body[2],
        -MAX_TORQUE_Z_NM,
        MAX_TORQUE_Z_NM,
    )

    # PhysxForceAPI is configured for world-frame values, so
    # convert the body-frame control torque into world space.
    desired_torque_world = (
        rotation_body_to_world @ desired_torque_body
    )

    # --------------------------------------------------------
    # APPLY FORCE AND TORQUE
    # --------------------------------------------------------

    cf2x_force_attr.Set(
        Gf.Vec3f(
            float(desired_force_world[0]),
            float(desired_force_world[1]),
            float(desired_force_world[2]),
        )
    )

    cf2x_torque_attr.Set(
        Gf.Vec3f(
            float(desired_torque_world[0]),
            float(desired_torque_world[1]),
            float(desired_torque_world[2]),
        )
    )

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    if print_timer_s >= PRINT_INTERVAL_S:
        print_timer_s = 0.0

        attitude_error_deg = np.degrees(
            attitude_error_body
        )

        print(
            "Position="
            f"({position_world[0]:.3f}, "
            f"{position_world[1]:.3f}, "
            f"{position_world[2]:.3f}) | "
            f"TargetZ={commanded_height:.3f} | "
            "Velocity="
            f"({filtered_linear_velocity_world[0]:.3f}, "
            f"{filtered_linear_velocity_world[1]:.3f}, "
            f"{filtered_linear_velocity_world[2]:.3f}) | "
            "AttErrorDeg="
            f"({attitude_error_deg[0]:.2f}, "
            f"{attitude_error_deg[1]:.2f}, "
            f"{attitude_error_deg[2]:.2f}) | "
            "Force="
            f"({desired_force_world[0]:.4f}, "
            f"{desired_force_world[1]:.4f}, "
            f"{desired_force_world[2]:.4f}) N | "
            "Torque="
            f"({desired_torque_world[0]:.6f}, "
            f"{desired_torque_world[1]:.6f}, "
            f"{desired_torque_world[2]:.6f}) Nm"
        )


# ============================================================
# INSTALL CONTROLLER
# ============================================================

cf2x_takeoff_subscription = (
    omni.physx.get_physx_interface()
    .subscribe_physics_step_events(cf2x_physics_step)
)

print("")
print("CF2X stable takeoff controller installed.")
print(f"Rigid body: {DRONE_BODY_PATH}")
print(f"Mass used: {controller_mass_kg:.6f} kg")
print(f"Target height: {TARGET_HEIGHT_M:.3f} m")
print(f"Takeoff speed: {TAKEOFF_SPEED_M_S:.3f} m/s")
print("Rotation axes are unlocked.")
print("Press Play to begin.")
