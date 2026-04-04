from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():

    # Models
    model_A = YOLO("yolov8n.pt")   # fast
    model_B = YOLO("yolov8x.pt")   # accurate

    video_path = "fixed.mp4"
    cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Cannot open video")
        return

    # Trackers
    tracker_A = DeepSort(max_age=30)
    tracker_B = DeepSort(max_age=30)

    VEHICLES = ["car", "motorcycle", "bus", "truck"]

    # Adjust this based on your video
    LINE_Y = 150
    OFFSET = 25

    counted_A = set()
    counted_B = set()

    prev_A = {}
    prev_B = {}

    total_A = 0
    total_B = 0

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("\nVideo complete")
            break

        frame_count += 1
        print(f"\n--- Frame {frame_count} ---")

       
        #MODEL A (YOLOv8n)
        
        results_A = model_A(frame, verbose=False)[0]
        detections_A = []

        if results_A.boxes is not None:
            for box in results_A.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model_A.names[cls]

                if label in VEHICLES and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections_A.append(([x1, y1, w, h], conf, label))

        tracks_A = tracker_A.update_tracks(detections_A, frame=frame)

        for track in tracks_A:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            #DRAW MODEL A
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"A-{track_id}",
                        (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            prev = prev_A.get(track_id, None)
            prev_y = prev[1] if prev else None

            if track_id not in counted_A:

                if prev_y is not None:
                    if prev_y < LINE_Y and cy >= LINE_Y:
                        total_A += 1
                        counted_A.add(track_id)
                        print(f"A COUNT → ID {track_id}")

                elif (LINE_Y - OFFSET) < cy < (LINE_Y + OFFSET):
                    total_A += 1
                    counted_A.add(track_id)
                    print(f"A COUNT (late) → ID {track_id}")

            prev_A[track_id] = (cx, cy)

       
        #MODEL B (YOLOv8x)
        results_B = model_B(frame, verbose=False)[0]
        detections_B = []

        if results_B.boxes is not None:
            for box in results_B.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model_B.names[cls]

                if label in VEHICLES and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections_B.append(([x1, y1, w, h], conf, label))

        tracks_B = tracker_B.update_tracks(detections_B, frame=frame)

        for track in tracks_B:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            #DRAW MODEL B
            cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
            cv2.putText(frame, f"B-{track_id}",
                        (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

            prev = prev_B.get(track_id, None)
            prev_y = prev[1] if prev else None

            if track_id not in counted_B:

                if prev_y is not None:
                    if prev_y < LINE_Y and cy >= LINE_Y:
                        total_B += 1
                        counted_B.add(track_id)
                        print(f"B COUNT → ID {track_id}")

                elif (LINE_Y - OFFSET) < cy < (LINE_Y + OFFSET):
                    total_B += 1
                    counted_B.add(track_id)
                    print(f"B COUNT (late) → ID {track_id}")

            prev_B[track_id] = (cx, cy)

        #DRAW COUNTING ZONE
        cv2.rectangle(frame,
                      (0, LINE_Y - OFFSET),
                      (frame.shape[1], LINE_Y + OFFSET),
                      (255, 0, 0), 2)

        #RUNNING TOTAL LOG
        print(f"Running Total → A: {total_A} | B: {total_B}")

        # DISPLAY
        cv2.putText(frame, f"A (n): {total_A}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"B (x): {total_B}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Tracking Accuracy System", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            print("\nStopped by user")
            break

    cap.release()
    cv2.destroyAllWindows()

    #FINAL REPORT
    print("\nFINAL REPORT")
    print(f"Model A (YOLOv8n): {total_A}")
    print(f"Model B (YOLOv8x): {total_B}")

    if total_B == 0:
        print("Cannot compute accuracy")
        return
    
    accuracy = (min(total_A, total_B) / max(total_A, total_B)) * 100
    difference = abs(total_A - total_B)

    print(f"📉 Difference: {difference}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
