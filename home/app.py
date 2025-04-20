import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Feature detection and stitching logic (your original code adapted here)
def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches


def trim(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    coords = np.argwhere(mask)
    if coords.size == 0:
        return frame
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return frame[x0:x1, y0:y1]


def stitch_images(images):
    base_img = images[0]
    for i in range(1, len(images)):
        img = images[i]
        gray1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp1, kp2, matches = detect_and_match_features(gray1, gray2)

        if matches is None or len(matches) < 10:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        if M is None:
            continue

        h1, w1 = base_img.shape[:2]
        h2, w2 = img.shape[:2]
        result = cv2.warpPerspective(base_img, M, (w1 + w2, h1))
        result[0:h2, 0:w2] = img

        center = (w2 // 2, h2 // 2)
        mask = np.ones_like(img[:, :, 0], dtype=np.uint8) * 255
        result = cv2.seamlessClone(img, result, mask, center, cv2.NORMAL_CLONE)

        base_img = trim(result)

    output_path = os.path.join(STATIC_FOLDER, "stitched_image.png")
    cv2.imwrite(output_path, base_img)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return redirect(url_for('index'))

    files = request.files.getlist('images')
    image_paths = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            image_paths.append(path)

    if len(image_paths) < 2:
        return "Need at least 2 images to stitch."

    images = [cv2.imread(path) for path in image_paths]
    stitched_image_path = stitch_images(images)

    return redirect(url_for('show_result'))


@app.route('/result')
def show_result():
    return render_template('result.html', image_url='/static/stitched_image.png')


if __name__ == "__main__":
    app.run(debug=True)
