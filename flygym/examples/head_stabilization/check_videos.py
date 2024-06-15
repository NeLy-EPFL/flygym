import cv2
import matplotlib.pyplot as plt
from pathlib import Path


excluded_videos = [
    # ("wave", "blocks", "train", "1.12_0.64"),
    # ("tripod", "blocks", "test", "1.14_0.58"),
]


def get_last_frame(video_file: Path):
    cap = cv2.VideoCapture(str(video_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    return frame[:, :, ::-1]


if __name__ == "__main__":
    base_dir = Path("./outputs/head_stabilization/random_exploration/")
    last_frames = {
        path.parent.name: get_last_frame(path) for path in base_dir.glob("*/*.mp4")
    }

    num_images = len(last_frames)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols

    # Save all last frames
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), tight_layout=True
    )
    for i, (title, frame) in enumerate(last_frames.items()):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(base_dir / "last_frames_all.png")

    # Save all last frames except excluded videos (because fly flipped)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), tight_layout=True
    )
    for i, (title, frame) in enumerate(last_frames.items()):
        gait, terrain, subset, _, dn_left, dn_right = title.split("_")
        dn_drives = dn_left + "_" + dn_right
        if (gait, terrain, subset, dn_drives) in excluded_videos:
            continue
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(base_dir / "last_frames_clean.png")
