from matching import _match
from ultralytics import YOLO
from track import Track
import cv2
import os

def main():

    # Load the model
    model = YOLO('yolov8n.pt') 

    sequence_path = "dataset/MOT15/train/TUD-Stadtmitte/img1"

    # Open the video source
    pattern = os.path.join(sequence_path, "%06d.jpg") # Selon le nommage
    cap = cv2.VideoCapture(pattern)
    # cap = cv2.VideoCapture(0) # 0 = webcam

    if not cap.isOpened():
        print("Can't find the video source")
        return

    tracks = [] # No tracks at the beginning
    next_ID = 1
    max_losses = 5  # number of frames without detection before deleting the track
    min_hits = 3    # number of detections before considering the track as confirmed
    frame_ID = 1
    display = True
    export_result = False

    if export_result:
        result_file = open(f"results/results.txt", 'w')

    # Loop over each frame of the video until we reach the end
    while cap.isOpened():
        # Update detections with YOLO
        success, frame = cap.read()

        if not success:
            break
        
        # Run detection on the frame
        # classes = 0 to detect only people
        # conf = 0.5 to avoid false detections
        results = model.predict(frame, classes = 0, conf = 0.5, verbose = False)[0]



        detections = []

        for box in results.boxes:
            # Get the coordinates x1, y1, x2, y2 of the box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Store the full BBox [x1, y1, x2, y2]
            detections.append([x1, y1, x2, y2])

        # Compute Kalman filter prediction
        for track in tracks:
            track.kf.predict()
        
        matches, unmatched_tracks, unmatched_detections = _match(tracks, detections)

        for track_idx, detection_idx in matches:
            tracks[track_idx].update(detections[detection_idx]) # Update KF and reset loss counter
        for track_idx in unmatched_tracks:
            tracks[track_idx].mark_missed() # Increment loss counter
        for detection_idx in unmatched_detections:
            # Create a new track
            new_track = Track(detections[detection_idx], next_ID)
            tracks.append(new_track)
            next_ID += 1

        tracks = [t for t in tracks if t.no_losses <= max_losses]

        # Draw ACTIVE tracks only
        for track in tracks:
            if track.hits >= min_hits:
                # Get the predicted BBox
                x1, y1, x2, y2 = track.get_bbox()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                if display:
                    # Draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display the ID above the BBox
                    cv2.putText(frame, f"ID {track.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if export_result:
                    width = x2 - x1
                    height = y2 - y1
                    res = f"{frame_ID},{track.id},{x1},{y1},{width},{height},1,-1,-1,-1\n"
                    result_file.write(res)
            
        if display:
            cv2.imshow("BasicTracker", frame)
            cv2.waitKey(1)
            
            # Close with window close button
            if cv2.getWindowProperty("BasicTracker", cv2.WND_PROP_VISIBLE) < 1:
                break
        frame_ID += 1

    if export_result:
        result_file.close()
    cap.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()