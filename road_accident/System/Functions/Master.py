import os
import re
import cv2
from datetime import datetime

from System.Controller.JsonEncoder import JsonEncoder
from System.Data.CONSTANTS import *
from System.Database.DatabaseConnection import DatabaseConnection
from System.Alerts.alert_manager import trigger_alert


class Master:
    def __init__(self):
        self.database = DatabaseConnection()

    # 🔹 Helper to sanitize district number
    def _parse_district(self, value):
        """Accepts int, '8', 'District 8', 'Dist-8', etc. Returns safe int."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            m = re.search(r'\d+', value)
            if m:
                return int(m.group(0))
        print(f"[Master] Warning: invalid district format ({value}); defaulting to 0")
        return 0

    def saveFrames(self, camera_id, starting_frame_id, frames, frame_width, frame_height):
        self.write(camera_id, frames, starting_frame_id, frame_width, frame_height, False)
        self.database.insertSavedFramesVid(camera_id, starting_frame_id)

    def write(self, camera_id, frames, starting_frame_id, frame_width, frame_height, is_crash=False):
        folder = "saved_crash_vid" if is_crash else "saved_frames_vid"
        file_path = './' + folder + '/' + "(" + str(camera_id) + ") " + str(starting_frame_id) + '.avi'
        if not os.path.exists(folder):
            os.makedirs(folder)
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              30, (frame_width, frame_height))

        for frame in frames:
            out.write(frame)
        print(f"camera_id_{camera_id}_{starting_frame_id}{folder} saved!{len(frames)}")
        out.release()

    def getVideoFrames(self, camera_id, frame_id, is_crash=False):
        folder = "saved_crash_vid" if is_crash else "saved_frames_vid"
        file_path = './' + folder + '/' + "(" + str(camera_id) + ") " + str(frame_id) + '.avi'
        cap = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def recordCrash(self, camera_id, starting_frame_id, crash_dimensions):
        new_frames = []
        from_no_of_times = PRE_FRAMES_NO

        while from_no_of_times >= 0:
            last_frames = from_no_of_times * 30
            new_frames_id = starting_frame_id - last_frames
            if new_frames_id > 0:
                new_frames.extend(self.getVideoFrames(camera_id, new_frames_id, False))
                frame_width = len(new_frames[0][0])
                frame_height = len(new_frames[0])
            from_no_of_times -= 1

        xmin, ymin, xmax, ymax = crash_dimensions[:4]
        if len(new_frames) > 60:
            no_of_frames = 3
        elif len(new_frames) > 30:
            no_of_frames = 2
        else:
            no_of_frames = 1

        if len(new_frames) >= 60:
            for i in range(len(new_frames) - 60, len(new_frames) - 30, 6):
                fill = -1
                cv2.rectangle(new_frames[i], (xmin, ymin), (xmax, ymax), (0, 0, 255), fill)
                cv2.putText(new_frames[i], "Crash!", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
            no_of_frames = 3

        for i in range(len(new_frames) - 30, len(new_frames), 1):
            fill = -1 if i % 2 == 0 else 2
            cv2.rectangle(new_frames[i], (xmin, ymin), (xmax, ymax), (0, 0, 255), fill)
            cv2.putText(new_frames[i], "Crash!", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)

        self.write(camera_id, new_frames, starting_frame_id, frame_width, frame_height, True)
        return no_of_frames

    def checkResult(self, camera_id, starting_frame_id, crash_dimentions, city, district_no):
        if len(crash_dimentions) == 0:
            return
        print("Sending Crash Has occured...")
        no_of_from_no = self.recordCrash(camera_id, starting_frame_id, crash_dimentions)
        self.database.insertCrashFramesVid(camera_id, starting_frame_id,
                                           PRE_FRAMES_NO + 1, city, district_no)
        self.sendNotification(camera_id, starting_frame_id, city, district_no)

        # ✅ Trigger Fire & Rescue alert safely
        if crash_dimentions:
            gps = None  # Example placeholder; integrate if you have coordinates
            try:
                safe_district = self._parse_district(district_no)
                trigger_alert(
                    event_name="Vehicle Collision",
                    city=str(city),
                    district=safe_district,
                    camera_id=str(camera_id),
                    frame_id=int(starting_frame_id),
                    crash_dims=crash_dimentions,
                    gps=gps,
                    snapshot_b64=None,
                )
            except Exception as e:
                print(f"[Master] trigger_alert() failed: {e}")

    def sendNotification(self, camera_id, starting_frame_id, city, district_no):
        jsonEncoder = JsonEncoder()
        time_now = str(datetime.utcnow().time()).split(".")[0]
        date = str(datetime.utcnow().date()) + " " + time_now
        crash_pic = self.getCrashPhoto(camera_id, starting_frame_id)
        jsonEncoder.sendNotification(camera_id, starting_frame_id, city, district_no, date, crash_pic)

    def executeQuery(self, start_date, end_date, start_time, end_time, city, district):
        dic_of_query = {}
        start_date_array = start_date.split("/")
        if len(start_date_array[0]) < 2:
            start_date_array[0] = "0" + start_date_array[0]
        if len(start_date_array[1]) < 2:
            start_date_array[1] = "0" + start_date_array[1]
        start_date = start_date_array[2] + "-" + start_date_array[1] + "-" + start_date_array[0]

        end_date_array = end_date.split("/")
        if len(end_date_array[0]) < 2:
            end_date_array[0] = "0" + end_date_array[0]
        if len(end_date_array[1]) < 2:
            end_date_array[1] = "0" + end_date_array[1]
        end_date = end_date_array[2] + "-" + end_date_array[1] + "-" + end_date_array[0]

        start_time += ":00"
        end_time += ":00"
        dic_of_query[START_DATE] = start_date
        dic_of_query[END_DATE] = end_date
        dic_of_query[START_TIME] = start_time
        dic_of_query[END_TIME] = end_time
        if city:
            dic_of_query[CITY] = city
        if district:
            dic_of_query[DISTRICT] = district

        results = self.database.selectCrashFramesList(dic_of_query)
        self.replyQuery(results)

    def replyQuery(self, results):
        list_data = []
        duplicates = {}
        for crash in results:
            camera_id, frame_id, city, district, crash_time = crash
            if camera_id in duplicates:
                continue
            crash_pic = self.getCrashPhoto(camera_id, frame_id)
            sending_msg = {
                CAMERA_ID: camera_id,
                STARTING_FRAME_ID: frame_id,
                CITY: city,
                DISTRICT: district,
                CRASH_TIME: crash_time,
                CRASH_PIC: crash_pic
            }
            list_data.append(sending_msg)
        jsonEncoder = JsonEncoder()
        jsonEncoder.replyQuery(list_data)

    def getCrashPhoto(self, camera_id, starting_frame_id):
        folder = "saved_crash_vid"
        file_path = './' + folder + '/' + "(" + str(camera_id) + ") " + str(starting_frame_id) + '.avi'
        cap = cv2.VideoCapture(file_path)
        if cap is None:
            return None
        total_frames = cap.get(7)
        frame_no = 89 if total_frames >= 90 else total_frames
        cap.set(1, frame_no)
        ret, photo = cap.read()
        return photo

    def sendVideoToGUI(self, camera_id, starting_frame_id):
        video_frames = self.getVideoFrames(camera_id, starting_frame_id, True)
        jsonEncoder = JsonEncoder()
        jsonEncoder.replyVideo(video_frames)

    def sendRecentCrashesToGUI(self):
        results = self.database.selectCrashFramesLast10()
        self.replyQuery(results)