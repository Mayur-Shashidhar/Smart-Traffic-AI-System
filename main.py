from ultralytics import YOLO
import cv2
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    model = YOLO("yolov8n.pt")

    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    tracker = DeepSort(max_age=30)

    VEHICLES = ["car", "motorcycle", "bus", "truck"]

    LINE_Y = 250
    OFFSET = 25

    counted_ids = set()
    prev_positions = {}   # stores (cx, cy)
    speeds = {}

    total_count = 0
    up_count = 0
    down_count = 0

    PIXEL_TO_METER = 0.05  # tuning factor

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("\nVideo complete")
            break

        frame_count += 1
        print(f"\n--- Frame {frame_count} ---")

        results = model(frame, verbose=False)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if label in VEHICLES and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    w = x2 - x1
                    h = y2 - y1

                    detections.append(([x1, y1, w, h], conf, label))

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            # SPEED CALCULATION
            prev = prev_positions.get(track_id, None)

            if prev is not None:
                px, py = prev

                distance_pixels = math.hypot(cx - px, cy - py)

                speed_mps = distance_pixels * fps * PIXEL_TO_METER
                speed_kmh = speed_mps * 3.6

                speeds[track_id] = int(speed_kmh)
            else:
                speeds[track_id] = 0

            # DRAW ID + SPEED
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"ID {track_id}",
                        (cx, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

            cv2.putText(frame, f"{speeds[track_id]} km/h",
                        (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)

            # FIXED prev_y extraction
            if prev is not None:
                prev_y = prev[1]
            else:
                prev_y = None

            # COUNTING + DIRECTION
            if track_id not in counted_ids:

                if prev_y is not None:

                    # DOWN
                    if prev_y < LINE_Y and cy >= LINE_Y:
                        total_count += 1
                        down_count += 1
                        counted_ids.add(track_id)

                        print(f"DOWN ID {track_id} Speed: {speeds[track_id]}")

                    # UP
                    elif prev_y > LINE_Y and cy <= LINE_Y:
                        total_count += 1
                        up_count += 1
                        counted_ids.add(track_id)

                        print(f"UP ID {track_id} Speed: {speeds[track_id]}")

                # Late detection
                elif (LINE_Y - OFFSET) < cy < (LINE_Y + OFFSET):
                    total_count += 1
                    down_count += 1
                    counted_ids.add(track_id)

                    print(f"DOWN (late) ID {track_id}")

            prev_positions[track_id] = (cx, cy)

        # DRAW ZONE
        cv2.rectangle(frame,
                      (0, LINE_Y - OFFSET),
                      (frame.shape[1], LINE_Y + OFFSET),
                      (255, 0, 0), 2)

        # DISPLAY COUNTS
        cv2.putText(frame, f"UP: {up_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"DOWN: {down_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"TOTAL: {total_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 3)

        cv2.imshow("Speed Estimation System", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()