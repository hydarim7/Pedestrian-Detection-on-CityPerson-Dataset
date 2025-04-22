!python /kaggle/working/yolov5/train.py \
    --img 1280 \
    --batch-size 16 \
    --epochs 10 \
    --data /kaggle/working/my_dataset1.yaml \
    --weights yolov5m6.pt \
    --cache

image_paths = [
    '/kaggle/working/yolov5/runs/detect/exp15/cologne_000118_000019_gtFinePanopticParts.jpg',
    '/kaggle/working/yolov5/runs/detect/exp15/dusseldorf_000025_000019_gtFinePanopticParts.jpg',
    '/kaggle/working/yolov5/runs/detect/exp15/dusseldorf_000025_000019_gtFinePanopticParts.jpg',
    '/kaggle/working/yolov5/runs/detect/exp15/erfurt_000042_000019_gtFinePanopticParts.jpg',


# Define grid dimensions: here we use 3 columns
cols = 3
rows = math.ceil(len(image_paths) / cols)

# Increase the overall figure size; adjust the multipliers to get larger images
fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
axs = axs.flatten()

for i, ax in enumerate(axs):
    if i < len(image_paths):
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.set_title(image_paths[i].split('/')[-1], fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')  # Hide any extra subplots

plt.tight_layout()
plt.show()

!python /kaggle/working/yolov5/val.py \
    --weights /kaggle/working/yolov5/runs/train/exp/weights/best.pt \
    --data /kaggle/working/my_dataset1.yaml \
    --img 1280 \
    --batch-size 64 \
    --conf-thres 0.3 \
    --iou-thres .5 \
    --task val \
    --save-json
