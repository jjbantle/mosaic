import numpy as np
import cv2
import os
import random
import scipy.misc

exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', '.jpe', '.jp2', '.tiff', '.tif', '.png']

def make_square(img):
    out = img.copy()

    if len(out.shape) == 3:
        ny, nx, nc = out.shape
    elif len(out.shape) == 2:
        ny, nx = out.shape
        nc = 0
    else:
        raise TypeError('Invalid image shape')
    
    if nx > ny:
        x0 = int(nx/2 - ny/2)
        x1 = int(nx/2 + ny/2)
        if nc > 0:
            out = out[:, x0:x1, :]
        else:
            out = out[:, x0:x1]
    elif ny > nx:
        y0 = int(ny/2 - nx/2)
        y1 = int(ny/2 + nx/2)
        if nc > 0:
            out = out[y0:y1, :, :]
        else:
            out = out[y0:y1, :]

    return out

def image_color(image):
    color = np.array((np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])))
    return color

def compare_color(color1, color2):
    r = np.sum((color1-color2)**2)**0.5
    return r

def crop_main(main, sub_dim):
    ny, nx, nc = main.shape
    
    y0 = (ny % sub_dim)/2
    y1 = ny - (ny % sub_dim + 1)/2
    
    x0 = (nx % sub_dim)/2
    x1 = nx - (nx % sub_dim + 1)/2
    
    return main[y0:y1, x0:x1, :]

def select_match(main_color, subs, sub_colors, sub_count, nSelect=10, alpha=10):
    color_diff = []
    for color, count in zip(sub_colors, sub_count):
        value = compare_color(main_color, color) + alpha*count
        color_diff.append(value)
    
    indices = range(len(color_diff))
    indices.sort(key=color_diff.__getitem__)
    
    subs = map(subs.__getitem__, indices)
    sub_colors = map(sub_colors.__getitem__, indices)
    sub_count = map(sub_count.__getitem__, indices)
    
    index = random.choice(range(nSelect))
    
    sub_count[index] += 1

    return subs[index], subs, sub_colors, sub_count

def get_subs(sub_files, sub_dim=64):
    subs = []
    sub_colors = []
    for i, filename in enumerate(sub_files):
        # read image
        image = cv2.imread(filename)
        # make square
        image = make_square(image)
        # resize
        subs.append(scipy.misc.imresize(image, size=(sub_dim, sub_dim, 3)))
        # get color
        sub_colors.append(image_color(image))
        
    return subs, sub_colors

def color_adjust(sub, target_color, tint=0.5):
    color = image_color(sub)
    delta = tint*(target_color - color)
    img = np.round(sub+delta)
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)
    
def apply_mask(image, mask, sub_dim):
    ny, nx, nc = image.shape
    large_mask = np.tile(mask, (ny/sub_dim, nx/sub_dim))
    large_mask3 = cv2.merge([large_mask, large_mask, large_mask])
    loc = np.where(large_mask3 == 0)
    result = image.copy()
    result[loc[0], loc[1], loc[2]] = 0
    return result

def circle_mask(dim):
    x, y = np.mgrid[:dim, :dim]
    circle = (x - dim/2) ** 2 + (y - dim/2) ** 2
    return circle < ((dim/2-1)**2)
    
def make_mosaic(main, sub_files, sub_dim=50, alpha=10, tint=0.0, nSelect=10):
    subs, sub_colors = get_subs(sub_files, sub_dim=sub_dim)
    sub_count = np.zeros(len(subs))

    # resize main image to divide evenly
    canvas = crop_main(main, sub_dim)

    # create canvas index
    ny, nx, nc = canvas.shape
    nSubs = (ny/sub_dim, nx/sub_dim)
    canvasMap = np.arange(nSubs[0]*nSubs[1]).reshape(nSubs)
    canvasMap = np.repeat(np.repeat(canvasMap, sub_dim, axis=0), sub_dim, axis=1)
    canvasMap3 = cv2.merge([canvasMap, canvasMap, canvasMap])
    mosaic = canvasMap3.copy()
      
    canvas_color = []
    for i in xrange(nSubs[0]*nSubs[1]):
        # get color for each tile in main image  
        loc = np.where(canvasMap3 == i)
        chip = canvas[loc[0], loc[1], loc[2]].reshape((sub_dim, sub_dim, 3))
        canvas_color = image_color(chip)
        # find the match for the canvas color
        sub, subs, sub_colors, sub_count = select_match(canvas_color, subs, sub_colors, sub_count, alpha=alpha, nSelect=nSelect)
        # adjust the colors if parameter set
        if tint > 0:
            sub = color_adjust(sub, canvas_color, tint)
        # insert the sub image into the mosaic
        mosaic[loc[0], loc[1], loc[2]] = sub.flat
    
    return mosaic

if __name__ == "__main__":
    infolder  = os.path.abspath(os.path.join(os.curdir, 'input'))
    outfolder = os.path.abspath(os.path.join(os.curdir, 'output'))
    
    for folder in os.listdir(infolder):
        print 'Working on case '+folder

        mainfolder = os.path.abspath(os.path.join(infolder, folder, 'main'))
        subfolder = os.path.abspath(os.path.join(infolder, folder, 'sub'))
        
        if not os.path.isdir(mainfolder):
            print 'Main image directory not found'
        if not os.path.isdir(subfolder):
            print 'Sub images directory not found'
        
        # get the main image
        mainfile = sorted(os.listdir(mainfolder))
        if len(mainfile) != 1:
            print 'Invalid number of main images - Using the first file.'
        name, ext = os.path.splitext(mainfile[0])
        if ext in exts:
            main = cv2.imread(os.path.join(mainfolder, mainfile[0]))
        
        # get the sub images
        subfiles = sorted(os.listdir(subfolder))
        if len(subfiles) < 1:
            print 'Invalid number of sub images'
        subs = []
        for filename in subfiles:
            name, ext = os.path.splitext(filename)
            if ext in exts:
                subs.append(os.path.join(subfolder, filename))
        
        result = make_mosaic(main, subs, sub_dim=64, tint=0.5, alpha=10, nSelect=10)
        cv2.imwrite(os.path.abspath(os.path.join(outfolder,folder+'.png')), result)