#!/usr/bin/env python3
"""
Simple video viewer for the tracked output.
Opens a web page to view the annotated video.
"""

import webbrowser
from pathlib import Path
from flask import Flask, send_file, render_template_string

app = Flask(__name__)

VIDEO_PATH = Path("output/tracked_color_h264.mp4").absolute()
FULL_VIDEO_PATH = Path("output/tracked_color_h264.mp4").absolute()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tracked Video Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            width: 100%;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 36px;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 18px;
        }

        .video-container {
            padding: 40px;
            background: #1a1a1a;
        }

        video {
            width: 100%;
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .info-panel {
            padding: 30px;
            background: #f7fafc;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .info-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .info-card h3 {
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }

        .info-card p {
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
        }

        .players-list {
            margin-top: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }

        .players-list h3 {
            color: #667eea;
            margin-bottom: 15px;
        }

        .player-tag {
            display: inline-block;
            background: #c6f6d5;
            color: #22543d;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-weight: 600;
            font-size: 14px;
        }

        .controls-info {
            background: #edf2f7;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .controls-info h4 {
            margin-bottom: 10px;
            color: #2d3748;
        }

        .controls-info ul {
            list-style: none;
            line-height: 1.8;
        }

        .controls-info li:before {
            content: "▶ ";
            color: #667eea;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Volleyball Tracking - Color-Based Filtering</h1>
            <p>Team Player Tracking with Auto-Detected Uniform Color</p>
        </div>

        <div class="video-container">
            <video controls autoplay muted>
                <source src="/video" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>

        <div class="info-panel">
            <div class="info-grid">
                <div class="info-card">
                    <h3>Resolution</h3>
                    <p>3840×2160</p>
                </div>
                <div class="info-card">
                    <h3>Duration</h3>
                    <p>6.7 minutes (full)</p>
                </div>
                <div class="info-card">
                    <h3>Total Frames</h3>
                    <p>12,030</p>
                </div>
                <div class="info-card">
                    <h3>Active Players</h3>
                    <p>~12 Players</p>
                </div>
            </div>

            <div class="players-list">
                <h3>📋 Tracking Info</h3>
                <p style="color: #2d3748; font-size: 16px; line-height: 1.6;">
                    <strong>Color-based filtering:</strong> Auto-learned uniform color HSV(25.0°, 42.5, 194.9) from first 100 frames.<br>
                    <strong>Filters applied:</strong> Size → Position → Color (±40° hue, ±60 saturation) → Sitting person.<br>
                    <strong>Result:</strong> 46,895 non-matching detections filtered out (~3.9 per frame), tracking only team players.
                </p>
            </div>

            <div class="controls-info">
                <h4>🎮 Video Controls:</h4>
                <ul>
                    <li>Click play button to start/pause</li>
                    <li>Use timeline to scrub through video</li>
                    <li>Click fullscreen button for better view</li>
                    <li>Adjust volume or mute as needed</li>
                    <li>Watch for green bounding boxes with player labels</li>
                </ul>
            </div>

            <div class="controls-info" style="background: #c6f6d5; margin-top: 15px;">
                <h4>✅ What You're Seeing:</h4>
                <ul>
                    <li><strong>Green boxes</strong> - Only team players wearing matching uniform color</li>
                    <li><strong>Labels</strong> - Player_[ID] (e.g., Player_123)</li>
                    <li><strong>Continuous tracking</strong> - Bounding boxes appear throughout entire 6.7 minute video</li>
                    <li><strong>Eliminated</strong> - Referees, coaches, opposing team, and spectators (different shirt colors)</li>
                    <li><strong>Smart filtering</strong> - Auto-learned dominant team color from first 100 frames</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the video player page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video')
def video():
    """Serve the video file."""
    if not VIDEO_PATH.exists():
        return "Video file not found", 404
    return send_file(str(VIDEO_PATH), mimetype='video/mp4')

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎬 VIDEO VIEWER STARTING")
    print("=" * 70)
    print(f"📹 Video: {VIDEO_PATH}")
    print(f"🌐 URL: http://127.0.0.1:9000")
    print("=" * 70)
    print("\n⏳ Opening browser...")

    # Open browser after short delay
    import threading
    import time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:9000')

    threading.Thread(target=open_browser, daemon=True).start()

    # Run server
    app.run(host='127.0.0.1', port=9000, debug=False)
