import numpy as np
import cv2
import os
from tqdm import tqdm

DIR = r""
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

input_path = r"" 
output_dir = os.path.join(DIR, "output")
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(DIR, "colorized_video.mp4")


print("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



def colorize_frame(frame):
    """Tek bir frame'i renkli hale getirir."""
    scaled = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

def process_image(image_path, output_dir):
    """Tek resim için colorization."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Resim bulunamadı: {image_path}")
    colorized = colorize_frame(image)
    out_path = os.path.join(output_dir, f"colorized{os.path.splitext(image_path)[1]}")
    cv2.imwrite(out_path, colorized)
    cv2.imshow("Original", image)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Colorized image saved at: {out_path}")

def process_video(video_path, output_video_path, fps_override=None):
    """Video için colorization (direct video output)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Colorizing video frames...")
    print("Colorizing video frames...")
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        colorized = colorize_frame(frame)
        out.write(colorized)

    cap.release()
    out.release()
    print(f"Colorized video saved at: {output_video_path}, total frames: {total_frames}")


ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
    process_image(input_path, output_dir)
elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
    process_video(input_path, output_video_path)
else:
    raise ValueError("Desteklenmeyen dosya türü. Sadece resim ve video destekleniyor.")
