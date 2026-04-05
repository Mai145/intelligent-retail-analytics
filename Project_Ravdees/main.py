import os
import cv2
import pandas as pd

# --- CONFIGURATION ---
VIDEO_DIR = "videos"
OUTPUT_DIR = "extracted_faces"
CSV_FILENAME = "dataset.csv"

# RAVDESS Filename Emotion Mapping
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def setup_environment():
    """Creates the output directory if it does not exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")


def process_videos():
    """Reads videos from subfolders, detects faces, crops them, and generates a CSV dataset."""
    setup_environment()

    # Initialize OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    dataset_records = []
    video_paths = []

    # Traverse through all subdirectories to find .mp4 files
    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if file.lower().endswith('.mp4'):
                # Save the full path of the video
                video_paths.append(os.path.join(root, file))

    total_videos = len(video_paths)

    if total_videos == 0:
        print(f"CRITICAL: Found 0 videos. Please ensure .mp4 files exist inside '{VIDEO_DIR}' or its subfolders.")
        return

    print(f"Found {total_videos} videos in '{VIDEO_DIR}' and subfolders. Starting extraction...")

    for index, video_path in enumerate(video_paths):
        # Extract filename to determine the emotion
        filename = os.path.basename(video_path)
        parts = filename.split('-')

        if len(parts) < 7:
            continue  # Skip files with invalid naming conventions

        emotion_code = parts[2]
        emotion_label = EMOTION_MAP.get(emotion_code, "Unknown")

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {filename}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select 3 frames from the middle of the video where the emotion is at its peak
        middle_frame = total_frames // 2
        frames_to_capture = [middle_frame - 5, middle_frame, middle_frame + 5]

        for frame_idx in frames_to_capture:
            if frame_idx < 0 or frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Face detection requires grayscale images
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # Iterate through detected faces
            for i, (x, y, w, h) in enumerate(faces):
                # Crop the face from the original frame
                face_crop = frame[y:y + h, x:x + w]
                # Resize to our model's standard input size
                face_resized = cv2.resize(face_crop, (224, 224))

                # Create a unique filename for the extracted face
                output_filename = f"{filename.replace('.mp4', '')}_f{frame_idx}_face{i}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                # Save the cropped face image
                cv2.imwrite(output_path, face_resized)

                # Add data to our records list
                dataset_records.append({
                    "image_path": output_path,
                    "emotion": emotion_label
                })
                break  # We only need one good face per frame

        cap.release()

        # Print progress update
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} / {total_videos} videos...")

    # Convert records to a Pandas DataFrame and save as CSV
    df = pd.DataFrame(dataset_records)
    df.to_csv(CSV_FILENAME, index=False)

    print("\n--- EXTRACTION COMPLETE ---")
    print(f"Total faces extracted and saved: {len(df)}")
    print(f"Dataset mapping saved to: {CSV_FILENAME}")


if __name__ == "__main__":
    process_videos()