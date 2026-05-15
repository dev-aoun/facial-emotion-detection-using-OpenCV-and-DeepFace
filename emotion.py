"""
COMPLETE FACIAL EMOTION RECOGNITION SYSTEM WITH PIE CHARTS
===========================================================

Project: Facial Emotion Recognition System
Real-time facial emotion detection using OpenCV and DeepFace

Features:
1. Real-time emotion detection from webcam
2. CSV data logging
3. Live statistics display
4. Beautiful UI with graphics
5. Color-coded emotions
6. Confidence scores
7. ROUND PIE CHARTS (saved as PNG)
8. Professional dashboard layout

Author: Muhammad Aoun
Email: maoun.778899@gmail.com
GitHub: https://github.com/dev-aoun
LinkedIn: https://www.linkedin.com/in/muhammad-aoun-128104373/

University: Bahria University Lahore Campus
Program: BSCS (Bachelor of Science in Computer Science)
Semester: 5-A
Course: Computer Science

Version: 1.0
Status: Production Ready
Date: May 2026

Description:
This is a complete facial emotion recognition system that implements real-time 
emotion detection from webcam video. It captures video frames, detects faces, 
analyzes emotions, and displays results with a professional dashboard interface.

Technologies Used:
- Python 3.8+
- OpenCV (cv2) - Computer Vision
- DeepFace - Deep Learning Emotion Analysis
- TensorFlow & Keras - Deep Learning Framework
- Matplotlib - Data Visualization
- NumPy & Pandas - Data Processing

License: MIT License (Open Source)

Acknowledgments:
- Bahria University Lahore Campus
- BSCS 5-A Faculty and Advisors
- OpenCV and DeepFace communities

For more information:
GitHub: https://github.com/dev-aoun
LinkedIn: https://www.linkedin.com/in/muhammad-aoun-128104373/
Email: maoun.778899@gmail.com

===========================================================
"""

import cv2
from deepface import DeepFace
import warnings
import numpy as np
from collections import defaultdict
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Configuration settings"""
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    EMOTION_PROCESS_INTERVAL = 5  # Process every 5 frames for performance
    
    EMOTION_COLORS = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),        # Blue
        'angry': (0, 0, 255),      # Red
        'surprise': (0, 255, 255), # Yellow
        'fear': (255, 0, 255),     # Magenta
        'disgust': (128, 0, 128),  # Purple
        'neutral': (200, 200, 200) # Gray
    }
    
    # Pie chart colors (RGB format for matplotlib)
    PIE_COLORS = {
        'happy': '#00FF00',        # Green
        'sad': '#FF0000',          # Red
        'angry': '#0000FF',        # Blue
        'surprise': '#FFFF00',     # Yellow
        'fear': '#FF00FF',         # Magenta
        'disgust': '#800080',      # Purple
        'neutral': '#C8C8C8'       # Gray
    }
    
    # UI Colors
    HEADER_COLOR = (0, 255, 200)      # Cyan
    TEXT_COLOR = (255, 255, 255)      # White
    ACCENT_COLOR = (255, 255, 0)      # Yellow
    BG_DARK = (10, 10, 10)            # Dark
    BG_PANEL = (20, 20, 20)           # Dark Gray

# ==================== UTILITY FUNCTIONS ====================

def create_csv_file():
    """Create CSV file for logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"emotion_log_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Frame', 'Emotion', 'Confidence', 'Faces', 'Timestamp'])
    
    return csv_filename, timestamp

def save_to_csv(csv_filename, frame_count, emotion, confidence, faces_count):
    """Save emotion data to CSV"""
    try:
        with open(csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                frame_count,
                emotion,
                f"{confidence:.4f}",
                faces_count,
                datetime.now().strftime("%H:%M:%S")
            ])
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def get_emotion_color(emotion):
    """Get color for emotion"""
    return Config.EMOTION_COLORS.get(emotion, Config.TEXT_COLOR)

def get_pie_color(emotion):
    """Get pie chart color for emotion"""
    return Config.PIE_COLORS.get(emotion, '#FFFFFF')

def print_statistics(frame_count, emotion_counts):
    """Print final statistics"""
    print("\n" + "="*70)
    print("EMOTION DETECTION SYSTEM - FINAL REPORT".center(70))
    print("="*70)
    print(f"\nSession Statistics:")
    print(f"  Total Frames Processed: {frame_count}")
    print(f"  Total Emotions Detected: {sum(emotion_counts.values())}")
    
    if emotion_counts:
        print(f"\n{'Emotion':<15} {'Count':<10} {'Percentage':<15} {'Bar Chart':<35}")
        print("-" * 70)
        
        total = sum(emotion_counts.values())
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {emotion:<13} {count:<10} {percentage:>6.2f}%        {bar}")
    
    print("\n" + "="*70)

# ==================== GRAPHICS FUNCTIONS ====================

def draw_header(frame):
    """Draw top header"""
    cv2.rectangle(frame, (0, 0), (Config.CAMERA_WIDTH, 60), Config.BG_DARK, -1)
    cv2.line(frame, (0, 60), (Config.CAMERA_WIDTH, 60), Config.HEADER_COLOR, 3)
    cv2.putText(frame, "REAL-TIME EMOTION DETECTION SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, Config.HEADER_COLOR, 2)

def draw_left_panel(frame, frame_count, faces_count, current_emotion, confidence_score):
    """Draw left side info panel"""
    info_x = 20
    info_y = 90
    
    # Background
    cv2.rectangle(frame, (10, 70), (400, 250), Config.BG_PANEL, 1)
    
    # Title
    cv2.putText(frame, "SESSION INFO", (info_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, Config.HEADER_COLOR, 2)
    cv2.line(frame, (info_x, info_y + 10), (380, info_y + 10), Config.HEADER_COLOR, 1)
    
    info_y += 40
    
    # Content
    cv2.putText(frame, f"Frame: {frame_count}",
                (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.TEXT_COLOR, 2)
    
    cv2.putText(frame, f"Faces: {faces_count}",
                (info_x, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    emotion_color = get_emotion_color(current_emotion)
    cv2.putText(frame, f"Emotion: {current_emotion.upper()}",
                (info_x, info_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)
    
    cv2.putText(frame, f"Confidence: {confidence_score:.2%}",
                (info_x, info_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.ACCENT_COLOR, 2)

def draw_statistics_panel(frame, emotion_counts):
    """Draw right side statistics panel"""
    stats_x = 870
    stats_y = 80
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (850, 0), (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT), Config.BG_PANEL, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    
    # Border
    cv2.rectangle(frame, (850, 70), (Config.CAMERA_WIDTH - 10, 650), Config.BG_PANEL, 1)
    
    # Title
    cv2.putText(frame, "STATISTICS", (stats_x, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, Config.HEADER_COLOR, 2)
    cv2.line(frame, (stats_x, stats_y + 15), (Config.CAMERA_WIDTH - 20, stats_y + 15), Config.HEADER_COLOR, 1)
    
    # Emotion breakdown with bar charts
    stats_y += 50
    total_emotions = sum(emotion_counts.values())
    
    for idx, (emotion, count) in enumerate(sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)):
        if idx >= 7:  # Show top 7 emotions
            break
        
        percentage = (count / total_emotions * 100) if total_emotions > 0 else 0
        color = get_emotion_color(emotion)
        
        # Emotion name
        text = f"{emotion[:8]:8} {percentage:5.1f}%"
        cv2.putText(frame, text, (stats_x, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, Config.TEXT_COLOR, 1)
        
        # Bar graph
        bar_width = int((percentage / 100) * 150)
        cv2.rectangle(frame, (stats_x + 120, stats_y - 10), (stats_x + 120 + bar_width, stats_y),
                      color, -1)
        cv2.rectangle(frame, (stats_x + 120, stats_y - 10), (stats_x + 120 + 150, stats_y),
                      (100, 100, 100), 1)
        
        stats_y += 30
    
    # Total count
    cv2.putText(frame, f"Total Detections: {total_emotions}",
                (stats_x, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.ACCENT_COLOR, 2)
    
    return frame

def draw_bottom_bar(frame, csv_filename):
    """Draw bottom info bar"""
    cv2.rectangle(frame, (0, 690), (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT), Config.BG_DARK, -1)
    cv2.line(frame, (0, 690), (Config.CAMERA_WIDTH, 690), Config.HEADER_COLOR, 2)
    
    cv2.putText(frame, f"File: {csv_filename}", (20, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.HEADER_COLOR, 1)
    
    cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (500, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.TEXT_COLOR, 1)
    
    cv2.putText(frame, "Press 'q' to quit", (1050, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

def draw_face_boxes(frame, faces, current_emotion, confidence_score):
    """Draw bounding boxes around faces with emotion labels"""
    emotion_color = get_emotion_color(current_emotion)
    
    for (x, y, w, h) in faces:
        # Main rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 3)
        
        # Corner circles
        for corner in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            cv2.circle(frame, corner, 5, emotion_color, 2)
        
        # Emotion label with background
        label = f"{current_emotion.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame, (x - 5, y - 45), (x + label_size[0] + 10, y - 5), emotion_color, -1)
        cv2.putText(frame, label, (x + 2, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        # Confidence bar under face
        bar_width = int(w * confidence_score)
        cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 15), emotion_color, -1)
        
        # Confidence percentage
        conf_text = f"{confidence_score:.0%}"
        cv2.putText(frame, conf_text, (x + w - 60, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

# ==================== PIE CHART GENERATION ====================

def generate_pie_charts(emotion_counts, timestamp):
    """Generate pie charts and save as PNG"""
    
    print("\n" + "="*70)
    print("GENERATING PIE CHARTS...")
    print("="*70)
    
    if not emotion_counts:
        print("No emotion data to generate charts")
        return
    
    # Prepare data
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    colors_list = [get_pie_color(e) for e in emotions]
    
    total = sum(counts)
    
    # ===== PIE CHART 1: Emotion Distribution =====
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors_list, startangle=90)
    plt.title('Emotion Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    
    pie_filename1 = f"emotion_distribution_pie_chart_{timestamp}.png"
    plt.savefig(pie_filename1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pie chart saved: {pie_filename1}")
    
    # ===== PIE CHART 2: Emotion Count (Donut Style) =====
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(counts, labels=emotions, autopct='%1.0f', 
                                         colors=colors_list, startangle=90)
    
    # Draw circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)
    
    plt.title('Emotion Count (Donut Chart)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    pie_filename2 = f"emotion_donut_chart_{timestamp}.png"
    plt.savefig(pie_filename2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Donut chart saved: {pie_filename2}")
    
    # ===== PIE CHART 3: Percentage with Legend =====
    fig, ax = plt.subplots(figsize=(12, 8))
    
    wedges, texts, autotexts = ax.pie(counts, colors=colors_list, startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    
    # Add percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    
    # Create legend with counts
    legend_labels = [f'{e}: {c} ({c/total*100:.1f}%)' for e, c in zip(emotions, counts)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    
    ax.set_title('Emotion Detection Results', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    pie_filename3 = f"emotion_results_pie_chart_{timestamp}.png"
    plt.savefig(pie_filename3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results pie chart saved: {pie_filename3}")
    
    # ===== PIE CHART 4: 3D Style Pie Chart =====
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    wedges, texts, autotexts = ax.pie(counts, labels=emotions, autopct='%1.1f%%',
                                        colors=colors_list, startangle=45,
                                        explode=[0.05] * len(emotions),
                                        shadow=True)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    ax.set_title('Emotion Analysis - Exploded Pie Chart', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    pie_filename4 = f"emotion_exploded_pie_chart_{timestamp}.png"
    plt.savefig(pie_filename4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exploded pie chart saved: {pie_filename4}")
    
    print("\n" + "="*70)
    print("PIE CHARTS GENERATED SUCCESSFULLY!")
    print("="*70)
    
    return [pie_filename1, pie_filename2, pie_filename3, pie_filename4]

# ==================== EMOTION DETECTION ====================

def analyze_face(face_roi):
    """Analyze emotion in face ROI"""
    try:
        result = DeepFace.analyze(
            face_roi,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        
        return emotion, confidence
    except Exception as e:
        return "Error", 0.0

# ==================== MAIN FUNCTION ====================

def main():
    """Main emotion detection function"""
    
    print("\n" + "="*70)
    print("FACIAL EMOTION RECOGNITION SYSTEM WITH PIE CHARTS".center(70))
    print("="*70)
    print("Author: Muhammad Aoun")
    print("Email: maoun.778899@gmail.com")
    print("GitHub: https://github.com/dev-aoun")
    print("LinkedIn: https://www.linkedin.com/in/muhammad-aoun-128104373/")
    print("University: Bahria University Lahore Campus")
    print("Program: BSCS 5-A")
    print("="*70)
    print("\nInitializing system...")
    print("Loading models...")
    
    # Initialize camera
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    
    # Create CSV file
    csv_filename, timestamp = create_csv_file()
    print(f"CSV file created: {csv_filename}")
    
    # Variables
    frame_count = 0
    current_emotion = "Analyzing..."
    confidence_score = 0.0
    emotion_counts = defaultdict(int)
    
    print(f"Camera initialized")
    print(f"Resolution: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}")
    print(f"FPS: {Config.CAMERA_FPS}")
    print("\n" + "="*70)
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Detect faces
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            frame_count += 1
            
            # Analyze emotions (every N frames for performance)
            if frame_count % Config.EMOTION_PROCESS_INTERVAL == 0:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    emotion, confidence = analyze_face(face_roi)
                    
                    if emotion != "Error":
                        current_emotion = emotion
                        confidence_score = confidence
                        emotion_counts[emotion] += 1
                        
                        # Save to CSV
                        save_to_csv(csv_filename, frame_count, emotion, confidence, len(faces))
            
            # ===== DRAW UI =====
            
            # Draw header
            draw_header(frame)
            
            # Draw left panel
            draw_left_panel(frame, frame_count, len(faces), current_emotion, confidence_score)
            
            # Draw right panel (statistics)
            frame = draw_statistics_panel(frame, emotion_counts)
            
            # Draw face boxes
            draw_face_boxes(frame, faces, current_emotion, confidence_score)
            
            # Draw bottom bar
            draw_bottom_bar(frame, csv_filename)
            
            # Display frame
            cv2.imshow('Facial Emotion Recognition System', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping emotion detection...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print_statistics(frame_count, emotion_counts)
        
        # Generate pie charts
        chart_files = generate_pie_charts(emotion_counts, timestamp)
        
        print(f"\nCSV file saved: {csv_filename}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total emotions detected: {sum(emotion_counts.values())}")
        
        print("\n" + "="*70)
        print("GENERATED FILES:")
        print("="*70)
        print(f"\nCSV Data File:")
        print(f"   • {csv_filename}")
        print(f"\nPie Charts (4 different styles):")
        if chart_files:
            for i, chart_file in enumerate(chart_files, 1):
                print(f"   {i}. {chart_file}")
        
        print("\n" + "="*70)
        print("Emotion detection completed successfully!".center(70))
        print("="*70 + "\n")
        print("Author: Muhammad Aoun")
        print("For more info: https://github.com/dev-aoun")
        print("="*70 + "\n")

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()