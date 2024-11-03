import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import math
import time
from pymavlink import mavutil
import threading

string_koneksi1 = 'udp:127.0.0.1:14551'
string_koneksi2 = 'tcp:127.0.0.1:5763'
konek = mavutil.mavlink_connection(string_koneksi1)
# konek.wait_heartbeat()

payload = r'D:\Kuliah 2.0\Lab APTRG\ultralytics-main\ultralytics-main\ultralytics\engine\runs\detect\train\weights\best.pt'
dropping = r'D:\Kuliah 2.0\Lab APTRG\ultralytics-main\ultralytics-main - Copy\ultralytics\engine\runs\detect\train\weights\best.pt'
exit_gate = r'D:\Kuliah 2.0\Lab APTRG\ultralytics-main\ultralytics-main - Copy (2)\ultralytics\engine\runs\detect\train\weights\best.pt'

tinggi = 0
ketinggian_lidar = 0
pick_executed = False
drop_executed = False

def arm_drone():
    konek.arducopter_arm()
    konek.motors_armed_wait()
    print("ARMING")

def disarm_drone():
    konek.arducopter_disarm()
    konek.motors_disarmed_wait()
    print("DISARMING")

def set_guided():
    print("mode : Guided")
    konek.mav.command_long_send(
        konek.target_system,                 
        konek.target_component,              
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 
        0,                                   
        1,                                   
        4,                                   
        0, 0, 0, 0, 0                        
    )
    time.sleep(3)

def takeoff(altitude):
    altitude = float(altitude)
    
    print("takeoff")
    konek.mav.command_long_send(
        konek.target_system, 
        konek.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)

def lidar_tembok():
    global tembok_lidar
    while True:
        msg = konek.recv_match(type='DISTANCE_SENSOR', blocking=True)
        a = msg.id
        if a == 6:
            tembok_lidar = msg.current_distance

def servo_magnet(servo_value, pwm_value):
    # Send servo command
    konek.mav.command_long_send(
        konek.target_system,
        konek.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,  # first transmission of this command
        servo_value,  # servo instance, offset by 8 MAIN outputs
        pwm_value,  # PWM pulse-width
        0,
        0,
        0,
        0,
        0,  # unused parameters
    )

def landing_pick():

    print("landing")
    konek.mav.command_long_send(
                konek.target_system,                 
                konek.target_component,              
                mavutil.mavlink.MAV_CMD_NAV_LAND,     
                0,                                    
                0, 0, 0, 0,                           
                0, 0, 0)
    print("wahana telah mendarat")
    
def baca_lidar():
    global ketinggian_lidar
    
    while True:
        msg = konek.recv_match(type='DISTANCE_SENSOR', blocking=True)
        
        id_ketinggian = msg.id
        if id_ketinggian == 0:
            ketinggian_lidar = msg.current_distance
        
def landing_pick_guided():
    global pick_executed
    print("Ketinggian : ", ketinggian_lidar)
    if ketinggian_lidar > 45:
        
        konek.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10, 
            konek.target_system,
            konek.target_component, 
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
            int(0b110111000111),  # Ignore position and use velocity (vx, vy, vz)
            0,  # x position (m), ignored
            0,  # y position (m), ignored
            0,  # z position (m), ignored
            0,  # vx velocity (m/s), no forward movement
            0,  # vy velocity (m/s), no lateral movement
            0.4,  # vz velocity (m/s), positive value to move down slowly
            0,  # x acceleration (m/s^2), ignored
            0,  # y acceleration (m/s^2), ignored
            0,  # z acceleration (m/s^2), ignored
            0,  # yaw, ignored
            0   # yaw_rate, ignored
        ))
        
    elif ketinggian_lidar < 45:
        landing_pick()
        time.sleep(5)
        disarm_drone()
        time.sleep(5)
        print("Arming servo")
        servo = 6
        pwm = 3000
        servo_magnet(servo, pwm)
        set_guided()
        time.sleep(5)
        arm_drone()
        time.sleep(5)
        takeoff(1.2)
        time.sleep(5)
        print("Continue dropping mission")
        pick_executed = True
    else:  # Continue descending until the altitude is 2 cm
        konek.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10, 
            konek.target_system,
            konek.target_component, 
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
            int(0b110111000111),  # Ignore position and use velocity (vx, vy, vz)
            0,  # x position (m), ignored
            0,  # y position (m), ignored
            0,  # z position (m), ignored
            0,  # vx velocity (m/s), no forward movement
            0,  # vy velocity (m/s), no lateral movement
            0,  # vz velocity (m/s), positive value to move down slowly
            0,  # x acceleration (m/s^2), ignored
            0,  # y acceleration (m/s^2), ignored
            0,  # z acceleration (m/s^2), ignored
            0,  # yaw, ignored
            0   # yaw_rate, ignored
        ))

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    return cap

def load_yolov8_model(model_path):
    model = YOLO(model_path)
    return model

def process_frame(frame, model, box_annotator):
    result = model(frame, agnostic_nms=True, verbose=False)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )
    return annotated_frame, detections

def control_drone_based_on_centroid(centroid_x, centroid_y):
    vertical_speeds = 0
    horizontal_speeds = 0

    # Check vertical movement first
    if centroid_y < 320:  # Object is above the center (move down)
        if centroid_y < 60:
            vertical_speeds = 0.25
        elif centroid_y < 80:
            vertical_speeds = 0.2
        elif centroid_y < 160:
            vertical_speeds = 0.2
        else:
            vertical_speeds = 0.1
    elif centroid_y > 400:  # Object is below the center (move up)
        if centroid_y > 450:
            vertical_speeds = -0.25
        elif centroid_y > 425:
            vertical_speeds = -0.2
        else:
            vertical_speeds = -0.1

    if 320 <= centroid_y <= 400:
        if centroid_x < 290:
            horizontal_speeds = -0.2  # Move left
        elif centroid_x > 350:
            horizontal_speeds = 0.2  # Move right

    return vertical_speeds, horizontal_speeds

# threading_tinggi = threading.Thread(target=baca_lidar)
# threading_tinggi.start()

def main():
    cap = initialize_camera()
    model = load_yolov8_model(payload)
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )
    
    tolerance = 30

    # arm_drone()
    # time.sleep(3)
    # takeoff(1.1)
    # time.sleep(1.5)

    detection_start_time = None
    no_detection_threshold = 3

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break
        
        annotated_frame, detections = process_frame(frame, model, box_annotator)

        frame_height, frame_width, _ = annotated_frame.shape

        cv2.circle(annotated_frame, (frame_width // 2, 350), tolerance, (255, 0, 0), 2)

        if len(detections) > 0:
            detection_start_time = time.time()
            for detection in detections:
                x1, y1, x2, y2 = detection[0]  # Directly access the bounding box coordinates
                w = x2 - x1
                h = y2 - y1
                centroid_x = int(x1 + w / 2)
                centroid_y = int(y1 + h / 2)

                vertical_speeds, horizontal_speeds = control_drone_based_on_centroid(centroid_x, centroid_y)

                print(f"Sending MAVLink command: vertical={vertical_speeds}, horizontal={horizontal_speeds}")
                
                konek.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                    10, konek.target_system, konek.target_component, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    int(0b110111000111), 0, 0, 0, vertical_speeds, horizontal_speeds, 0, 0, 0, 0, 0, 0
                ))

                if 290 <= centroid_y <= 415 and 265 <= centroid_x <= 380:
                    if not pick_executed:
                        print("Payload aligned for picking up")
                        landing_pick_guided()
                        
                    elif not drop_executed:
                        print("Dropping object")
                        print("Disarming servo")
                        servo = 6
                        pwm = 100
                        servo_magnet(servo, pwm)
                        time.sleep(1)
                        drop_executed = True

                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                cv2.line(annotated_frame, (centroid_x, centroid_y), (frame_width // 2, 350), (0, 255, 0), 2)

        else:
            if detection_start_time is None:
                detection_start_time = time.time()

            elapsed_time = time.time() - detection_start_time

            if elapsed_time > no_detection_threshold:
                print("No object detected for 5 seconds. Moving forward.")
                
                konek.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                    10, konek.target_system, konek.target_component, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    int(0b110111000111), 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0
                ))

        cv2.imshow("yolov8", annotated_frame)

        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
