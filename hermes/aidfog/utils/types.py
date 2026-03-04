from dataclasses import dataclass
from enum import Enum
from collections import deque
from multiprocessing import Queue
from multiprocessing.synchronize import Event as _Event
from multiprocessing.sharedctypes import Synchronized
import can
import struct
from typing import Callable, Optional
import numpy as np


@dataclass
class ServoReference:
    position: float
    velocity: float
    acceleration: float


@dataclass
class ServoImpedanceGains:
    position: float
    velocity: float
    acceleration: float


@dataclass
class ServoParameters:
    position_min: float
    position_max: float
    velocity_min: float
    velocity_max: float
    current_min: float
    current_max: float
    torque_min: float
    torque_max: float
    Kt_nominal: float  # from the TMotor website (actually 1/Kvll)
    current_factor: float
    Kt_actual: float
    gear_ratio: float
    num_pole_pairs: int
    is_use_derived_torque_constants: bool


class ServoMotorEnum(Enum):
    AK10_9 = ServoParameters(
        position_min=-32000,                    # -3200 deg
        position_max=32000,                     # 3200 deg
        velocity_min=-100000,                   # -100000 rpm electrical speed
        velocity_max=100000,                    # 100000 rpm electrical speed
        current_min=-1500,                      # -60A is the acutal limit but set to -15A
        current_max=1500,                       # 60A is the acutal limit but set to 15A
        torque_min=-15,                         # NM
        torque_max=15,                          # NM
        Kt_nominal=0.16,                        # from TMotor website (actually 1/Kvll)
        current_factor=0.59,                    # ! UNTESTED CONSTANT
        Kt_actual=0.921914 * 0.16,              # ! TESTED CONSTANT adjusted using torque sensor
        gear_ratio=9.0,
        num_pole_pairs=21,
        is_use_derived_torque_constants=False,  # `true` if you have a better model
    )
    AK80_9 = ServoParameters(
        position_min=-32000,                    # -3200 deg
        position_max=32000,                     # 3200 deg
        velocity_min=-32000,                    # -32000 rpm electrical speed
        velocity_max=32000,                     # 32000 rpm electrical speed
        current_min=-1500,                      # -60A is the acutal limit but set to -15A
        current_max=1500,                       # 60A is the acutal limit but set to 15A
        torque_min=-30,                         # NM
        torque_max=30,                          # NM
        Kt_nominal=0.091,                       # from TMotor website (actually 1/Kvll)
        current_factor=0.59,
        Kt_actual=0.115,
        gear_ratio=9.0,
        num_pole_pairs=21,
        is_use_derived_torque_constants=False,  # `true` if you have a better model
    )
    AK80_8 = ServoParameters(
        position_min=-32000,                    # -3200 deg
        position_max=32000,                     # 3200 deg
        velocity_min=-32000,                    # -32000 rpm electrical speed
        velocity_max=32000,                     # 32000 rpm electrical speed
        current_min=-1500,                      # -60A is the acutal limit but set to -15A
        current_max=1500,                       # 60A is the acutal limit but set to 15A
        torque_min=-25,                         # NM
        torque_max=25,                          # NM
        Kt_nominal=0.199,                       # from TMotor website (actually 1/Kvll)
        current_factor=0.59,
        Kt_actual=0.69 * 0.199,
        gear_ratio=8.0,
        num_pole_pairs=21,
        is_use_derived_torque_constants=False,  # `true` if you have a better model
    )


class ServoErrorCodes(Enum):
    NO_ERR = 0                  # No Error
    OVER_TEMP = 1               # Over temperature fault
    OVER_CURR = 2               # Over current fault
    OVER_VOLT = 3               # Over voltage fault
    UNDER_VOLT = 4              # Under voltage fault
    ENCODER_ERR = 5             # Encoder fault
    PHASE_IMBALANCE_ERR = 6     # Phase current unbalanced fault (The hardware may be damaged)


class ServoCanPacketEnum(Enum):
    CAN_PACKET_SET_DUTY = 0             # Motor runs in duty cycle mode
    CAN_PACKET_SET_CURRENT = 1          # Motor runs in current loop mode
    CAN_PACKET_SET_CURRENT_BRAKE = 2    # Motor current brake mode operation
    CAN_PACKET_SET_RPM = 3              # Motor runs in current loop mode
    CAN_PACKET_SET_POS = 4              # Motor runs in position loop mode
    CAN_PACKET_SET_ORIGIN_HERE = 5      # Set origin mode
    CAN_PACKET_SET_POS_SPD = 6          # Position velocity loop mode


@dataclass
class BatteryData:
    timestamp: float
    temperature: float
    voltage: float
    current: float
    power: float


@dataclass
class ServoMotorData:
    timestamp: float
    position: float
    velocity: float
    current: float
    temperature: int
    error: bytes


class StateEnum:
    class Idle(Enum):
        IDLE = 0

    class Walking(Enum):
        IDLE = 1
        WALKING = 2

    class SitToStand(Enum):
        STANCE = 3
        LOWERING = 4
        SITTING = 5
        RISING = 6

    class StairAscent(Enum):
        STANCE = 7
        SWING = 8
        PUSH_OFF = 9

    class StairDescent(Enum):
        DOUBLE_SUPPORT = 10
        SWING = 11
        STANCE = 12


@dataclass
class StateTransition:
    timestamp: float
    state: int


@dataclass
class ModeTransition:
    timestamp: float
    mode: int
    sequence_id: int


@dataclass
class PhaseEstimate:
    timestamp: float
    phase: float


@dataclass
class MotorCommand:
    motor_id: str
    timestamp: float
    data: list[bytes]
    control_mode: int


@dataclass
class ModeContext:
    bus: can.BusABC
    K: dict[str, ServoImpedanceGains]
    motor_latest_data: dict[int, deque[ServoMotorData | None]]
    fatigue: "Synchronized[float]"
    mode_changed_queue: "Queue[ModeTransition]"
    state_changed_queue: "Queue[StateTransition]"
    phase_estimate_queue: "Queue[PhaseEstimate]"
    motor_command_queue: "Queue[MotorCommand]"
    is_stop_new_data_event: _Event


@dataclass
class ModeTuple:
    id: int
    text: str


class ModeEnum(Enum):
    IDLE = ModeTuple(id=0, text="idle")
    WALKING = ModeTuple(id=1, text="walking")
    SIT_TO_STAND = ModeTuple(id=2, text="sit_to_stand")
    STAIR_ASCENT = ModeTuple(id=3, text="stair_ascent")
    STAIR_DESCENT = ModeTuple(id=4, text="stair_descent")


class StairLeadingLegEnum(Enum):
    NONE = 0
    RIGHT = 1
    LEFT = 2


class WalkingFirstStrideEnum(Enum):
    NONE = 0
    ONGOING = 1
    TAKEN = 2
    DEACTIVATED = 3


@dataclass
class StairAscentParameters:
    # Transition thresholds
    stance_to_swing_th_gyr: float
    stance_to_swing_th_roll: float
    stance_to_swing_kn_roll: float
    swing_to_pushoff_gyr_min: float
    swing_to_pushoff_gyr_max: float
    swing_to_pushoff_th_roll: float
    pushoff_trigger_delay: float
    pushoff_to_stance_gyr_min: float
    pushoff_to_stance_gyr_max: float
    pushoff_to_stance_th_roll: float
    pushoff_to_stance_roll_small: float
    pushoff_to_stance_kn_roll: float

    # Action timing and impedance control
    imp_gain_rise_time: float
    gain_step: float
    pushoff_kn_threshold: float

    # Torque scaling
    torque_scale: float
    torque_norm_angle: float

    # Inactivity detection
    inactivity_gyr_threshold: float
    inactivity_time_step: float

    # Movement detection
    movement_angle_threshold: float
    movement_gyr_threshold: float
    movement_sum_threshold: float


@dataclass
class StairDescentParameters:
    # Transition thresholds
    double_to_swing_th_gyr: float
    stance_th_gyr_min: float
    stance_th_gyr_max: float
    stance_kn_roll: float
    stance_to_swing_th_gyr: float
    stance_to_swing_kn_roll: float
    swing_to_double_gyr_min: float
    swing_to_double_gyr_max: float
    swing_to_double_th_roll: float
    swing_to_double_inactive_dur: float
    stance_to_double_gyr_min: float
    stance_to_double_gyr_max: float
    stance_to_double_th_roll: float
    stance_to_double_inactive_dur: float

    # Action parameters
    angle_activation_threshold: float
    torque_knee: float
    torque_hip: float
    ramp_time: float
    gain_step: float

    # Inactivity detection
    inactivity_gyr_threshold: float
    inactivity_time_step: float

    # Movement detection
    movement_angle_threshold: float
    movement_gyr_threshold: float
    movement_sum_threshold: float


@dataclass
class SitToStandParameters:
    # Transition thresholds
    stance_to_lowering_th_gyr: float
    stance_to_lowering_th_roll: float
    stance_to_lowering_torso_roll: float
    stance_to_lowering_kn_roll: float
    lowering_to_sitting_th_gyr_min: float
    lowering_to_sitting_th_gyr_max: float
    lowering_to_sitting_phase_threshold: float
    sitting_to_rising_th_gyr: float
    sitting_to_rising_phase_threshold: float
    sitting_to_rising_idle_dur_min: float
    rising_to_stance_th_gyr_min: float
    rising_to_stance_th_gyr_max: float
    rising_to_stance_kn_roll: float
    rising_to_stance_phase_threshold: float
    reset_phase_threshold: float
    inactivity_angle_threshold: float

    # Impedance gains and timing
    imp_gain_rise_time: float
    gain_step: float
    active_ramp_time: float

    # Phase calculation parameters
    phase_start_angle: float
    phase_end_angle: float


@dataclass
class WalkingParameters:
    inactivity_gyr_threshold: float
    inactivity_idle_transition_time: float
    traj_shift: tuple[float, float, float, float]
    inactivity_time_step: float
    first_stride_end_gyr: float
    reset_phase_threshold: float
    inactivity_angle_threshold: float


@dataclass
class ExoNiclaMapping:
    torso: str
    thigh_right: str
    thigh_left: str
    shank_right: str
    shank_left: str
    # foot_right: str
    # foot_left: str


@dataclass
class MaskParsingTuple:
    mask: int
    format: str
    key: str
    num_bytes: int


class NiclaI2cCommand(Enum):
    CMD_START = 0xF0
    CMD_SEND = 0xAA
    CMD_STOP = 0x0F
    CMD_DEFAULT = 0x00
    CMD_DEFAULT_REPLY = 0xFF


class NiclaConnectionType(Enum):
    BLE = 0
    I2C = 1


class NiclaPacketMask(Enum):
    ACC     = MaskParsingTuple(mask=0x01, format="3h", key="acceleration", num_bytes=6)
    GYR     = MaskParsingTuple(mask=0x02, format="3h", key="gyroscope", num_bytes=6)
    MAG     = MaskParsingTuple(mask=0x04, format="3h", key="magnetometer", num_bytes=6)
    EULER   = MaskParsingTuple(mask=0x08, format="3f", key="euler", num_bytes=12)
    QUAT    = MaskParsingTuple(mask=0x10, format="4f", key="quaternion", num_bytes=16)
    TEMP    = MaskParsingTuple(mask=0x20, format="f", key="temperature", num_bytes=4)
    BARO    = MaskParsingTuple(mask=0x40, format="f", key="pressure", num_bytes=4)
    HUM     = MaskParsingTuple(mask=0x80, format="f", key="humidity", num_bytes=4)


@dataclass
class NiclaData:
    timestamp: int
    sequence_id: int
    acceleration: Optional[tuple[int, int, int]] = None
    gyroscope: Optional[tuple[int, int, int]] = None
    magnetometer: Optional[tuple[int, int, int]] = None
    euler: Optional[tuple[float, float, float]] = None
    quaternion: Optional[tuple[float, float, float, float]] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None

    @classmethod
    def from_bytes(cls, data: bytearray):
        mask, timestamp, sequence_id = struct.unpack_from("<BII", data)
        kwargs = {"timestamp": timestamp, "sequence_id": sequence_id}
        offset: int = 9

        for modality in NiclaPacketMask:
            if mask & modality.value.mask:
                val = struct.unpack_from(modality.value.format, data, offset)
                kwargs[modality.value.key] = val[0] if len(val) == 1 else val
                offset += modality.value.num_bytes
        return cls(**kwargs)


@dataclass
class NiclaNumpyGetter:
    func: Callable[[list[NiclaData]], np.ndarray]


class NiclaDataGetMethods(Enum):
    acceleration = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.acceleration, data)), dtype=np.int16
        )
    )
    gyroscope = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.gyroscope, data)), dtype=np.int16
        )
    )
    magnetometer = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.magnetometer, data)), dtype=np.int16
        )
    )
    euler = NiclaNumpyGetter(
        func=lambda data: np.array(list(map(lambda n: n.euler, data)), dtype=np.float32)
    )
    quaternion = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.quaternion, data)), dtype=np.float32
        )
    )
    temperature = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.temperature, data)), dtype=np.float32
        )
    )
    pressure = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.pressure, data)), dtype=np.float32
        )
    )
    humidity = NiclaNumpyGetter(
        func=lambda data: np.array(
            list(map(lambda n: n.humidity, data)), dtype=np.float32
        )
    )


class NiclaPayloadMode:
    def __init__(
        self,
        is_acc: bool,
        is_gyr: bool,
        is_mag: bool,
        is_euler: bool,
        is_quat: bool,
        is_temp: bool,
        is_baro: bool,
        is_hum: bool,
    ):
        self._data_getters: dict[str, Callable[[list[NiclaData]], np.ndarray]] = {}

        if is_acc:
            self._data_getters[NiclaDataGetMethods.acceleration.name] = (
                NiclaDataGetMethods.acceleration.value.func
            )
        if is_gyr:
            self._data_getters[NiclaDataGetMethods.gyroscope.name] = (
                NiclaDataGetMethods.gyroscope.value.func
            )
        if is_mag:
            self._data_getters[NiclaDataGetMethods.magnetometer.name] = (
                NiclaDataGetMethods.magnetometer.value.func
            )
        if is_euler:
            self._data_getters[NiclaDataGetMethods.euler.name] = (
                NiclaDataGetMethods.euler.value.func
            )
        if is_quat:
            self._data_getters[NiclaDataGetMethods.quaternion.name] = (
                NiclaDataGetMethods.quaternion.value.func
            )
        if is_temp:
            self._data_getters[NiclaDataGetMethods.temperature.name] = (
                NiclaDataGetMethods.temperature.value.func
            )
        if is_baro:
            self._data_getters[NiclaDataGetMethods.pressure.name] = (
                NiclaDataGetMethods.pressure.value.func
            )
        if is_hum:
            self._data_getters[NiclaDataGetMethods.humidity.name] = (
                NiclaDataGetMethods.humidity.value.func
            )

    def get_data_getters(self) -> dict[str, Callable[[list[NiclaData]], np.ndarray]]:
        return self._data_getters


@dataclass
class ExoMotorMapping:
    hip_right: str
    hip_left: str
    knee_right: str
    knee_left: str
