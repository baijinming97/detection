from utils import *
model = YOLO('WoundYoloSeg.pt', verbose=False)
root = "D:\Projects\Wound\data"
tif_files = list_tif_paths(root)
for i in tqdm(range(len(tif_files))):
    impath = tif_files[i]
    h, w = cv2.imread(impath).shape[:2]
    extract = model([impath])[0].masks
    if extract!=None:
        mask = extract.data.cpu().numpy()[0].astype(np.uint8)
        mask = mask*255
        mask = cv2.resize(mask,(w,h))
        create_json(mask, impath, h, w, erosion_size=150, grid_spacing=100)
