from utils import *
root = "D:\Projects\Wound\data"
jpg_files = list_jpg_paths(root)
for i in tqdm(range(len(jpg_files))):
    jpg_file = jpg_files[i]
    jpg_im = cv2.imread(jpg_file)
    tif_file = jpg_file.replace('jpg', 'tif')
    cv2.imwrite(tif_file, jpg_im)
    os.remove(jpg_file)