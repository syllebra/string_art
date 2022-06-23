import os
import random
import numpy as np
import cv2
from skimage.draw import line, line_aa
import numpy as np
from tqdm import tqdm
import math
from joblib import Parallel, delayed

from point_cloud import PointCloud

class PinLayout():
    # Pins are layout in a ractangular shape around the canvas frame (with some small random offsets to avoid string artifacts)
    RECTANGLE = 0
    # Pins are layout in a circualr shape around the image
    CIRCLE = 1
    # Pins are layout around a perimeter defined by a separate mask
    PERIM_IMAGE = 2
    # Pins are layout in a full point cloud across the entire image (or in a given mask)
    POINT_CLOUD = 3

###########################################################
# INPUT PARAMETERS SECTION
###########################################################

# Source image path (relative or absolute)
source_path = 'sources/boat_00.jpg'

# Layout type (cf above)
TYPE = PinLayout.POINT_CLOUD

# Number of pins (excluding point cloud mode)
NB_PINS = 300 

# "Weight" of each rendered string (1-255)
LINE_WEIGHT = 47#37

# For all modes except point cloud, the number of pins around the
# last picked where it is not possible to go directly
SPACING = NB_PINS // 10

# Total number of iterations (strings) to draw
ITERATIONS = 60000

# Export a render every N images
SAVE_EVERY = 80

# If true, draws white string on black background, other way around otherwise
INVERT = False

# Scale ratio for input (leave auto for coherent parameters across the algorithm)
auto_scale_ratio = True
# if auto_scale_ratio is False, scale ratio for the residual computation
scale_ratio = 1

# Out put ratio (rendering dimension with respect to input image)
out_ratio = 1.4

# Debug image display ratio with respect to rendering final dimensions
display_ratio = 1

# For point cloud mode, the average radius, in pixels, between two close pins
POINT_CLOUD_AVERAGE_RADIUS = 16

# if point cloud mode, path to mask where pins will not be layed out
# (can be None for full image point cloud layout)
point_cloud_mask = None#'sources/skull_00_mask.png'

# for perimeter mode, path to the image of perimeter (black and white)
perimeter_path = None


# Early stop parameters (experimental)
ACTIVATE_EARLY_STOP = False
EARLY_STOP_MEAN_THRESHOLD = 10
EARLY_STOP_CONSECUTIVE = 500 // POINT_CLOUD_AVERAGE_RADIUS

###########################################################



# Load an image in grayscale
img = cv2.imread(source_path,cv2.IMREAD_GRAYSCALE)

if(auto_scale_ratio):
    mini = min(img.shape[0],img.shape[1])
    scale_ratio = 600/mini

#img = 255-auto_canny(img)
img = cv2.resize(img,(0,0),fx=scale_ratio,fy=scale_ratio, interpolation=cv2.INTER_LANCZOS4)


W = img.shape[1]
H = img.shape[0]
print(img.shape)

###########################################################
# orb = cv2.ORB_create(nfeatures=2000)
# kp, des = orb.detectAndCompute(img, None)

# kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

# cv2.imshow('ORB', kp_img)
# cv2.waitKey()

###########################################################

def pins_square(rdm=0.4):
    # Create pins positions
    tot_perim_len = 2 * (img.shape[0] + img.shape[1])
    pxdec = tot_perim_len / NB_PINS

    cur = np.array([0,0], np.float32)
    dir = 0
    pins = np.zeros((NB_PINS,2), np.int32) # y, x
    for i in range(NB_PINS):
        pins[i,:] = np.array([cur[1],cur[0]])
        if(dir == 0):
            cur[0] = cur[0] + pxdec + int(pxdec * (random.random()-0.5)*2 * rdm)
            if(cur[0]>=W):
                cur[1] = cur[1] + cur[0] - W
                cur[0] = W-1
                dir = 1
        elif(dir == 1):
            cur[1] = cur[1] + pxdec + int(pxdec *(random.random()-0.5)*2 * rdm)
            if(cur[1]>=H):
                cur[0] = cur[0] - (H-cur[1])
                cur[1] = H-1
                dir = 2
        elif(dir == 2):
            cur[0] = cur[0] - pxdec + int(pxdec *(random.random()-0.5)*2 * rdm)
            if(cur[0]<0):
                cur[1] = cur[1] + cur[0]
                cur[0] = 0
                dir = 3
        elif(dir == 3):
            cur[1] = cur[1] - pxdec + int(pxdec *(random.random()-0.5)*2 * rdm)

    # finish in case full perimeter not finished
    add = []
    while(cur[1]>0):
        cur[1] = cur[1] - pxdec + int(pxdec *(random.random()-0.5)*2 * rdm)
        if(cur[1]>0):
            add.append([cur[1],cur[0]])

    if(len(add)>0):
        add = np.array(add, np.int32)
        pins = np.concatenate((pins,add), axis = 0)

    # for i in range(NB_PINS):
    #     pins[i][0] += int((random.random()-0.5)*2 * rdm)
    #     pins[i][1] += int((random.random()-0.5)*2 * rdm)

    return pins

def pins_circle(mult=1.0):
    # Create pins positions
    
    radius = min(W,H) * 0.5 * mult
    inc = math.pi *2.0 / NB_PINS
    pins = np.zeros((NB_PINS,2), np.int32) # y, x
    center = np.array([H*0.5,W*0.5])
    for i in range(NB_PINS):
        pins[i,:] = center + np.array([math.cos(i*inc),math.sin(i*inc)])*radius

    return pins

def pins_point_cloud(av_rad=20, mask_path=None):
    mask = None
    if(mask_path is not None):
        mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(0,0),fx=scale_ratio,fy=scale_ratio, interpolation=cv2.INTER_LANCZOS4)
        mask=cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=av_rad//4)

    num_pts = int(2.5*(img.shape[0]*img.shape[1]) / (math.pi*av_rad*av_rad))
    pc = PointCloud(num_pts*2)
    # pc.scatterOnMask(img, num_pts, av_rad//2, threshold=0.0)
    # pc.compute_cache_K_closest(k=60)
    # pc.relax(iterations=40, lock_dist=img.shape[0]/30, W = img.shape[1], H = img.shape[0], radius = av_rad*1.1)
    pc.create_random_from_precomputed(img.shape[1],img.shape[0], av_rad)

    if(mask is not None):
        pc.mask(mask)

    pc.compute_cache_K_closest(k=60)

    return pc.p[:pc.count], pc

def pins_perimeter(perimeter_image):
    perim = cv2.imread(perimeter_image,cv2.IMREAD_GRAYSCALE)
    perim = cv2.resize(perim,(img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    perim = cv2.threshold(perim, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(perim, 1,cv2.CHAIN_APPROX_NONE)
    
    length = 0
    for c in contours:
        length += cv2.arcLength(c, True)
        print(c.shape)
    print("Length:",length)

    inc = length / float(NB_PINS)
    pins = np.zeros((NB_PINS,2), np.int32) # y, x
    num = 0
    for c in contours:
        lp = c[0][0]
        left = inc
        for p in c:
            nextp = p[0]
            travel = np.linalg.norm(nextp-lp)
            if(travel >= left):
                ratio = (travel-left) / inc
                pin_pos = lp + (nextp-lp) * ratio
                pins[num] = [pin_pos[1],pin_pos[0]]
                num += 1
                left = inc - (travel-left)
            else:
                left -= travel
            lp = nextp

        print(c[0][0])
    
    if(num<NB_PINS):
        pins = pins[:num]

    # dbg = img.copy()
    # if(len(img.shape)<=2):
    #     dbg = dbg//3
    #     dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
    # for c in contours:
    #     col =  list(np.random.random(size=3) * 256)
    #     cv2.drawContours(dbg, [c], -1, col, 2)

    # for (i,p) in enumerate(pins):
    #     cv2.circle(dbg, (p[1],p[0]), 1, (255,0,255),-1)

    # cv2.imshow("thres", dbg)
    # cv2.waitKey(0)
    # exit(0)

    return pins, perim

in_mask = None
pins = None
pc = None
if(TYPE == PinLayout.RECTANGLE):
    pins = pins_square()
elif(TYPE == PinLayout.CIRCLE):
    pins = pins_circle(1.3)
elif(TYPE == PinLayout.PERIM_IMAGE):
    if(perimeter_path is not None):
        pins, in_mask = pins_perimeter(perimeter_path)
elif(TYPE == PinLayout.POINT_CLOUD):
    pins, pc = pins_point_cloud(av_rad=POINT_CLOUD_AVERAGE_RADIUS, mask_path=point_cloud_mask)

NB_PINS = len(pins)
print("Total pins:",NB_PINS)
if(pc is None):
    pc = PointCloud(pins.shape[0])
    pc.p = pins
    pc.count = pins.shape[0]

def find_best_starting_pin(debug = False):
    target_val = np.max(img) if INVERT else np.min(img)
    pixel_candidates = np.array(np.where(img == target_val)).T
    print(pixel_candidates.shape)
    tmp = None
    if(debug):
        tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if (pc is None):
        return None

    if(debug):
        for i in range(pins.shape[0]):
            cv2.circle(tmp, (int(pins[i,1]),int(pins[i,0])), 1, (0,255,255), -1)

    ids = []
    dist = []
    for i in range(pixel_candidates.shape[0]):
        id, dis = pc.closestPoint(pixel_candidates[i])
        ids.append(id)
        dist.append(dis)
        if(debug):
            cv2.circle(tmp, (int(pixel_candidates[i,1]),int(pixel_candidates[i,0])), 3, (255,0,255), 1)
            cv2.line(tmp, (int(pixel_candidates[i,1]),int(pixel_candidates[i,0])), (int(pins[id,1]),int(pins[id,0])), (0,0,255),1)

    id = np.argmin(np.array(dis))
    id = ids[id]

    if(debug):
        cv2.circle(tmp, (int(pins[id,1]),int(pins[id,0])), 3, (255,255,0), 1)
        cv2.imshow("Low",tmp)
        cv2.waitKey(0)

    return id

def line_ids(p0,p1, width = None, height = None, antialiased = False, mask = None):
    if(width is None): width = W
    if(height is None): height = H

    wr = width/W
    hr = height/H
    start = (int(p0[0]*wr),int(p0[1]*hr))
    end = (int(p1[0]*wr),int(p1[1]*hr))
    # being start and end two points (x1,y1), (x2,y2)
    
    discrete_line = list(zip(*line_aa(*start, *end))) if(antialiased) else list(zip(*line(*start, *end)))
    discrete_line = np.array(discrete_line).T
    
    ids = np.where((discrete_line[1]>=0) & (discrete_line[1]<width-1) & (discrete_line[0]>=0) & (discrete_line[0]<height-1))
    ids = ids[0]
    if(len(ids) == 0):
        return None, None
    discrete_line = discrete_line[:,ids]
    coords = discrete_line[0:2,:].astype(np.int32)
    vals = discrete_line[2,:] if (antialiased) else np.array(1)

    if (mask is not None):
        # check if inside masked shape (for example in concave perimeter shapes)
        if(coords is None or len(coords)==0):
            return None, None
        values = mask[coords[0],coords[1]]
        sm = np.mean(values)
        if(sm < 254):
            return None, None

    return coords, vals

def par_line_ids(i, j, width = None, height = None, antialiased = False, mask = None):
    ln, vals = line_ids(pins[i],pins[j], width, height, antialiased, mask)
    return (i,j), ln, vals

def compute_line_ids_cache(width = None, height = None, antialiased = False, mask = None):
    if(width is None): width = W
    if(height is None): height = H

    cache = {}
    if(TYPE == PinLayout.POINT_CLOUD):
        for (i,nb) in tqdm(pc.cache_K_closest.items()):
            a = [int(pc.p[i][1]),int(pc.p[i][0])]
            for p2 in nb:
                b = [int(pc.p[p2][1]),int(pc.p[p2][0])]
                ln, vals = line_ids(pins[i],pins[p2], width, height, antialiased, mask)
                if(ln is not None):
                    cache[(i,p2)] = {"ln":ln,"vals":vals}
    else:    
        for i in tqdm(range(len(pins))):
            results = Parallel(n_jobs=3)(delayed(par_line_ids)(i,j,width, height, antialiased, mask) for j in range(i+1,len(pins)))
            tmp = {x[0]: {"ln":x[1],"vals":x[2]} for x in results}
            cache.update(tmp)
        
    return cache

print("Computing line caches...")
orig_cache = compute_line_ids_cache(antialiased = True, mask = in_mask)
out_cache = compute_line_ids_cache(width = W*out_ratio, height = H * out_ratio, antialiased = True)

error = np.zeros((H,W), np.int32)
error[:,:]=255
error = error - img
if(INVERT):
    error = 255 - error

dbg = np.zeros((int(H*out_ratio),int(W*out_ratio),3), np.int32)
dbg[:,:,:]= 0 if INVERT else 255

# dbg = Image.new('RGBA', size = (img.shape[1], img.shape[0]), color = (255, 255, 255, 255))
# draw = ImageDraw.Draw(dbg)

def get_cached_line(c,i,j):
    if((i,j) in c):
        v = c[(i,j)]
        return v["ln"], v["vals"]
    if((j,i) in c):
        v = c[(j,i)]
        return v["ln"], v["vals"]
    return None, None


def par_search_best(i, pin, last, cache, error):
    to_test = (pin+SPACING+i)%NB_PINS
    
    if(to_test == pin or to_test == last): return None
    ln, vals = get_cached_line(cache,to_test,pin)
    if(ln is None or len(ln)==0):
        return None
    values = error[ln[0],ln[1]]
    sm = np.mean(values)
    return sm

def render(iterations=ITERATIONS, history_dir = "./steps", parallel = 0):

    # looking for steps save directory
    save_dir=None
    basename=None
    if(history_dir is not None):
        basename,_ = os.path.splitext(os.path.basename(source_path))
        cpt=0
        while True:
            save_dir = os.path.join(history_dir,basename,("run_%03d" % cpt))
            if (not os.path.exists(save_dir)):
                break
            cpt += 1
        os.makedirs(os.path.abspath(save_dir))
        print("Saving directory:",os.path.abspath(save_dir))


    early_stop_cpt = 0
    num_jumps = 0

    pin = find_best_starting_pin()# np.random.choice(NB_PINS)
    last = pin
    for l in tqdm(range(iterations)):
        #pin = np.random.choice(NB_PINS)
        sel = None
        max = -math.inf
        best = -1
        test = pin
        
        if(parallel == 0):
            for i in range(NB_PINS-SPACING*2):
                sm = par_search_best(i, pin, last, orig_cache, error)
                if(sm is None):
                    continue
                if(sm>max):
                    max = sm
                    best = (pin+SPACING+i)%NB_PINS
        else:
            sms = Parallel(n_jobs=parallel, prefer="threads")(delayed(par_search_best)(ii, pin, last, orig_cache, error) for ii in range(NB_PINS-SPACING*2))
            sms = np.array(sms, dtype=np.float64)
            if(np.count_nonzero(np.isfinite(sms))>0):
                max = np.max(sms)
                best = np.nanargmax(sms)
                best = (pin+SPACING+best)%NB_PINS

        if(ACTIVATE_EARLY_STOP and max<EARLY_STOP_MEAN_THRESHOLD):
            early_stop_cpt += 1
            if(early_stop_cpt>=EARLY_STOP_CONSECUTIVE):
                print("Mean threshold reached. Early stopping...")
            break
        else:
            early_stop_cpt = 0

        if(best == -1):
            num_jumps += 1
            print("Total jumps: ", num_jumps)
            pin = np.random.choice(NB_PINS)
            last = pin
            continue

        p0 = pins[pin]
        p1 = pins[best]

        ln, vals = get_cached_line(orig_cache,best,pin)
        upscaled, vu = get_cached_line(out_cache,best,pin)
        # if(upscaled is None):
        #     upscaled, vu = line_ids(p0*out_ratio,p1*out_ratio, width = W*out_ratio, height = H*out_ratio)
        #     print("pb")
            
        last = best

        if(ln is not None and upscaled is not None):
            torem = (vu * LINE_WEIGHT).astype(np.int32)
            if(INVERT):
                torem = -torem
            dbg[upscaled[0],upscaled[1],0] -= torem
            dbg[upscaled[0],upscaled[1],1] -= torem
            dbg[upscaled[0],upscaled[1],2] -= torem
            dbg[dbg<0]=0
            dbg[dbg>255]=255
        #cv2.line(dbg,(p0[1]*out_ratio,p0[0]*out_ratio),(p1[1]*out_ratio,p1[0]*out_ratio), (0,0,0), 1, 8)

        sel,vs = get_cached_line(orig_cache,best,pin)
        if(sel is not None):
            error[sel[0],sel[1]] -= (vs * LINE_WEIGHT).astype(np.int32)
        #error[error<0]=0
        pin = best

        if(l %1 == 0):
            cv2.imshow('residual',cv2.resize(255-(error.astype(np.uint8)),(0,0),fx=display_ratio,fy=display_ratio,interpolation=cv2.INTER_CUBIC))
            cv2.imshow('StringArt',cv2.resize(dbg.astype(np.uint8),(0,0),fx=display_ratio,fy=display_ratio,interpolation=cv2.INTER_CUBIC))
            key = cv2.waitKey(10)
            if(key == 32): # space pause
                cv2.waitKey(0)
            if(key == 27 or key =='q'): # "escape" to quit"
                break
        
        if(history_dir is not None and (l % SAVE_EVERY == 0)):
            name = basename+("_%07d" % l)+".png"
            name = os.path.join(save_dir,name)
            cv2.imwrite(name,dbg)
    print("Total jumps: ", num_jumps)
# cv2.imshow('StringArt',cv2.resize(dbg.astype(np.uint8),(0,0),fx=display_ratio,fy=display_ratio,interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)

render(parallel=0)

for (i,p) in enumerate(pins):
    cv2.circle(dbg, (int(p[1]*out_ratio),int(p[0]*out_ratio)), 1, (255,0,255),-1)

cv2.imshow('StringArt',cv2.resize(dbg.astype(np.uint8),(0,0),fx=display_ratio,fy=display_ratio,interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()