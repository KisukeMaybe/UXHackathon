import time
import board
import busio
import struct
import math
import usb_hid
import digitalio
from adafruit_hid.mouse import Mouse

# --- I2C setup ---
i2c = busio.I2C(board.GP21, board.GP20)  # SCL, SDA
MPU_ADDR = 0x68

# --- Built-in LED setup ---
led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

# --- Mode Switch Setup ---
mode_switch = digitalio.DigitalInOut(board.GP18)
mode_switch.direction = digitalio.Direction.INPUT
mode_switch.pull = digitalio.Pull.UP  # Pull high; reads False when connected to GND

# --- Helper functions for raw register access ---
def write_reg(addr, value):
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(MPU_ADDR, bytes([addr, value]))
    finally:
        i2c.unlock()

def read_regs(addr, n):
    buf = bytearray(n)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(MPU_ADDR, bytes([addr]))
        i2c.readfrom_into(MPU_ADDR, buf)
    finally:
        i2c.unlock()
    return buf

# Wake up MPU6050
write_reg(0x6B, 0x00)
time.sleep(0.1)

# Set accel config to ±2g (default)
write_reg(0x1C, 0x00)

# --- Mouse setup ---
mouse = Mouse(usb_hid.devices)

def read_gyro_raw():
    """Read gyro X,Y,Z in deg/sec"""
    data = read_regs(0x43, 6)
    gx, gy, gz = struct.unpack(">hhh", data)
    return gx / 131.0, gy / 131.0, gz / 131.0

# --- Calibration ---
CALIBRATION_DURATION = 2.0
print("Calibrating... hold still for 2 seconds")

cal_start = time.monotonic()
samples_gx, samples_gy, samples_gz = [], [], []

while time.monotonic() - cal_start < CALIBRATION_DURATION:
    # Blink LED rapidly during calibration
    led.value = int((time.monotonic() - cal_start) * 10) % 2 == 0
    gx, gy, gz = read_gyro_raw()
    samples_gx.append(gx);  samples_gy.append(gy);  samples_gz.append(gz)
    time.sleep(0.02)

n = len(samples_gx)
gyro_offset = [sum(samples_gx)/n, sum(samples_gy)/n, sum(samples_gz)/n]
print(f"Calibration done. Offsets: {[round(v,2) for v in gyro_offset]}")

def read_gyro():
    gx, gy, gz = read_gyro_raw()
    return gx - gyro_offset[0], gy - gyro_offset[1], gz - gyro_offset[2]

# --- Cursor & Inversion Settings ---
SENSITIVITY_X = -0.9
SENSITIVITY_Y = -0.6
MOUSE_SENSITIVITY_X = 15 * SENSITIVITY_X
MOUSE_SENSITIVITY_Y = 15 * SENSITIVITY_Y
CURSOR_SMOOTHING = 0.3
ACCEL_EXPONENT = 1.8

# Inversion Logic
invert_multiplier = 1
last_toggle_time = time.monotonic()
TOGGLE_INTERVAL = 10.0 

# Mining detection settings
THRESHOLD = 180
COOLDOWN = 0.2
WINDOW = 1.0
TARGET_RATE = 2
mine_times = []
last_trigger = 0
mouse_down = False

# Mode tracking
CURSOR_MODE = mode_switch.value 
previous_switch_state = CURSOR_MODE
last_time = time.monotonic()
last_mouse_x = 0.0
last_mouse_y = 0.0

def apply_accel(value):
    return math.copysign(abs(value) ** ACCEL_EXPONENT, value)

def update_led():
    led.value = CURSOR_MODE

update_led()

while True:
    now = time.monotonic()
    
    # --- 10-Second Inversion Toggle ---
    if now - last_toggle_time >= TOGGLE_INTERVAL:
        invert_multiplier *= -1
        last_toggle_time = now
        print(f"DIRECTIONS FLIPPED! Multiplier: {invert_multiplier}")
        
        # Blink LED to signal the change
        led.value = not led.value
        time.sleep(0.05)
        led.value = not led.value

    gx, gy, gz = read_gyro()
    
    # --- Physical Mode Toggle ---
    current_switch_state = mode_switch.value
    if current_switch_state != previous_switch_state:
        CURSOR_MODE = current_switch_state
        previous_switch_state = current_switch_state
        update_led()
        print(f"Mode -> {'CURSOR' if CURSOR_MODE else 'MINE'}")
        
        if mouse_down:
            mouse.release(Mouse.LEFT_BUTTON)
            mouse_down = False
        last_mouse_x, last_mouse_y = 0.0, 0.0
        mine_times = []
        time.sleep(0.05) 

    # --- MINE MODE ---
    if not CURSOR_MODE:
        magnitude = math.sqrt(gx*gx + gy*gy + gz*gz)
        if magnitude > THRESHOLD and (now - last_trigger) > COOLDOWN:
            mine_times.append(now)
            last_trigger = now

        mine_times = [t for t in mine_times if now - t < WINDOW]
        rate = len(mine_times)

        if rate > TARGET_RATE and not mouse_down:
            mouse.press(Mouse.LEFT_BUTTON)
            mouse_down = True
        elif rate <= TARGET_RATE and mouse_down:
            mouse.release(Mouse.LEFT_BUTTON)
            mouse_down = False

    # --- CURSOR MODE ---
    else:
        # Smoothing and Raw Input
        raw_x = gy * CURSOR_SMOOTHING + last_mouse_x * (1 - CURSOR_SMOOTHING)
        raw_y = -gx * CURSOR_SMOOTHING + last_mouse_y * (1 - CURSOR_SMOOTHING)
        last_mouse_x, last_mouse_y = raw_x, raw_y

        # Movement Calculation with Inversion
        move_x = int(apply_accel(raw_x) * MOUSE_SENSITIVITY_X / 100)
        move_y = int(apply_accel(raw_y) * MOUSE_SENSITIVITY_Y / 100)

        if move_x != 0 or move_y != 0:
            mouse.move(
                x=(-1 * move_x) * invert_multiplier, 
                y=(move_y) * invert_multiplier
            )

    time.sleep(0.01)
