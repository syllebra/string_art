

import random
import numpy as np
import math
from tqdm import tqdm
import os

def halton(b, num):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    for i in range(num):
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

def get_points(nb, W, H):
    points = np.zeros((nb,2), np.float32)
    x = [x * W for x in halton(2,nb)]
    y = [x * H for x in halton(3,nb)]
    points[:,0] = y
    points[:,1] = x
    return points

class PointCloud(object):
    
    def __init__(self, pre_alloc = 2000, precomputed_path="pc_precomputed.npz"):
        self.p = np.empty((pre_alloc,2), np.float32)
        self.count = 0
        self.cache_K_closest = {}
        self.cache_K_closest_last_k = None

        self.precomputed = None
        if(os.path.exists(precomputed_path)):
            self.precomputed = np.load(precomputed_path, allow_pickle=True)
    
    def add(self, pt):
        self.p[self.count] = pt
        self.count += 1

    def create_random_from_precomputed(self, W, H, R=20):
        if(self.precomputed is None):
            return
        pts = self.precomputed["points"]
        Ws = self.precomputed["W"]
        Hs = self.precomputed["H"]
        Rs = self.precomputed["radius"]

        R_ratio = Rs/R
        W_tmp = W*R_ratio
        H_tmp = H*R_ratio

        offset = np.array([random.random() * (Hs - H_tmp), random.random() * (Ws - W_tmp)])
        p = pts - offset
        ids = (np.where((p[:,0]>0) & (p[:,1]>0) & (p[:,0]<H_tmp)  & (p[:,1]<W_tmp)))
        p = p[ids]

        p = p / np.array([R_ratio,R_ratio])
        self.p = p
        self.count = self.p.shape[0]
        

    def scatterOnMask(self, maskImg, numPoints, min_dist, threshold = 0.2):
        #inspired from https://github.com/hooyah/nailedit/blob/master/PointCloud.py
        print ('scattering ',numPoints, ' points')

        if(threshold == 0):
            self.p = get_points(numPoints, maskImg.shape[1], maskImg.shape[0])
            self.count = self.p.shape[0]
            return
        
        # brute force
        num = 0
        fail = 0

        f = maskImg.flatten()
        interesting = np.where(f >= threshold*255)[0]
        np.random.shuffle(interesting)
        for i in interesting:
            x = float(i % maskImg.shape[1])
            y = float(i / maskImg.shape[1])
            pt = np.array([y,x])
            if self.count==0 or self.closestPoint(pt)[1] >= min_dist:
                self.add(pt)
                num += 1
                if num >= numPoints:
                    break
            else:
                fail += 1
                if fail >= numPoints*20:
                    break

        print ("successfully scattered", num, "of", numPoints, "points")

    def compute_cache_K_closest(self, max_radius=None, k=None):
        self.cache_K_closest = {}
        self.cache_K_closest_last_k = k

        if(k is not None):
            k = k+1

        for i in tqdm(range(self.count)):
            ids, _ = self.closestPoints(self.p[i],max_radius=max_radius, k=k)
            if(len(ids)>0):
                self.cache_K_closest[i] = ids
            else:
                print(i)

    def update_cache(self):
        self.compute_cache_K_closest(k=self.cache_K_closest_last_k)

    def closestPoint(self, to):
        dst = self.p[:self.count] - to
        dst = np.sum(dst * dst, axis = 1)
        closest = np.nanargmin(dst)
        return closest, math.sqrt(dst[closest])

    def closestPoints(self, to, max_radius=None, k=None):
        dst = self.p[:self.count] - to
        dst = np.sum(dst * dst, axis = 1)

        ids = np.argsort(dst)
        sort_d2 = dst[ids]
        if(max_radius is not None):
            sort_d2 = sort_d2[sort_d2<=max_radius*max_radius]
            if(sort_d2 is None or sort_d2.shape[0]==0):
                return None, None
        
        sz = sort_d2.shape[0]
        if(k is not None):
            sz = min(sz, k)
        
        ids = ids[1:sz]

        return ids, np.sqrt(dst[ids])

    

    def relax(self, ratio = 0.25, iterations = 100, dist = 20, lock_dist=-1, W=-1, H=-1, radius = None):
        import cv2
        dbg = np.zeros((H,W,3), np.uint8)

        for iter in tqdm(range(iterations)):
            pts = np.copy(self.p)
            ids = np.arange(self.count)
            np.random.shuffle(ids)
            for i in ids:
                if(i not in self.cache_K_closest): print(i); continue
#                nb=self.cache_K_closest[i]
            # for (i,nb) in self.cache_K_closest.items():
                mult = 1.0
                if(lock_dist>=0):
                    if(self.p[i][0]<lock_dist):
                        mult = self.p[i][0]/lock_dist
                    if(self.p[i][0]>H - lock_dist):
                        mult = (H-self.p[i][0])/lock_dist
                    if(self.p[i][1]<lock_dist):
                        mult = self.p[i][1]/lock_dist
                    if(self.p[i][1]>W - lock_dist):
                        mult = (W-self.p[i][1])/lock_dist
                    mult = min(1.0,mult)
                    mult = max(0.0,mult)

                ci, _ = self.closestPoints(self.p[i], k = self.cache_K_closest_last_k, max_radius=radius)
                # farther = self.p[ci[np.argmax(dists)]]
                # pts[i] += (farther - self.p[i])*ratio*mult
                # for tmp, d in zip(ci, dists):
                #     pts[tmp] += (pts[tmp] - self.p[i])*ratio*mult*0.2
                if(ci is not None and len(ci)>0):
                    mean = np.mean(self.p[ci], axis =0)
                    pts[i] += (self.p[i]-mean)*ratio*mult
            self.p = np.copy(pts)

            dbg[:,:,:] = 0
            for p in self.p[:self.count]:
                cv2.circle(dbg, (int(p[1]),int(p[0])), 1, (255,0,255),-1)
            # for (i,nb) in pc.cache_K_closest.items():
            #     a = [int(pc.p[i][1]),int(pc.p[i][0])]
            #     for p2 in nb:
            #         b = [int(pc.p[p2][1]),int(pc.p[p2][0])]
            #         cv2.line(dbg, a, b, (0,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Generating Point Cloud", dbg)
            cv2.waitKey(10)

        self.update_cache()

    def mask(self, mask):
        ret = self.p[:self.count]
        inside = []
        for i in range(ret.shape[0]):
            pt = ret[i,:].astype(np.int32)
            if(pt[0]>=0 and pt[0]<mask.shape[0] and pt[1]>=0 and pt[1]<mask.shape[1]):
                if(mask[pt[0],pt[1]]>128):
                    inside.append(i)
        self.p = self.p[inside]
        self.count = self.p.shape[0]
        if(len(self.cache_K_closest) != 0):
            self.update_cache()


    def clean(self, radius, W, H):
        torem = set()
        changed = set()

        for i in tqdm(range(self.count)):
            if(i in torem): continue
            if(self.p[i][0]<0 or self.p[i][0]>=H or self.p[i][1]<0 or self.p[i][1]>=W):
                torem.add(i)
            else:
                ids, dist = self.closestPoints(self.p[i], max_radius=radius)
                if(ids is None or ids.shape[0]<=0): continue

                mean = (np.sum(self.p[ids], axis =0) + self.p[i]) / (ids.shape[0]+1)
                self.p[i] = mean
                changed.add(i)

                for id in ids:
                    torem.add(id)

        left = [x for x in range(self.count) if x not in torem]
        self.p = self.p[left]
        self.count = self.p.shape[0]

        print("Removed", len(torem)," After:",self.count)
        if(len(self.cache_K_closest) != 0):
            self.update_cache()

if __name__ == "__main__":
    
    # pc.p = [np.array([24,3],np.float32),np.array([26,3],np.float32),np.array([4,3],np.float32)]
    # print(pc.closestPoints(np.array([10,10]), k = 3, max_radius=17))

    import cv2
    #img = cv2.imread("sources/einstein_00.jpg", cv2.IMREAD_GRAYSCALE)
    img = np.zeros((768,1024), np.uint8)

    radius = 5
    num_pts = int(2.5*(img.shape[0]*img.shape[1]) / (math.pi*radius*radius))

    pc = PointCloud(num_pts*2)

    while True:
        pc.create_random_from_precomputed(img.shape[1],img.shape[0], 10)
        print(pc.count)
        # pc.scatterOnMask(img, num_pts, radius//2, threshold=0.0)
        # print(pc.p.shape)
        # pc.compute_cache_K_closest(k=60)
        # pc.relax(iterations=100, lock_dist=img.shape[0]/30, W = img.shape[1], H = img.shape[0], radius = radius*1.1)

        dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for p in pc.p[:pc.count]:
            cv2.circle(dbg, (int(p[1]),int(p[0])), 1, (255,0,255),-1)
        for (i,nb) in pc.cache_K_closest.items():
            a = [int(pc.p[i][1]),int(pc.p[i][0])]
            for p2 in nb:
                b = [int(pc.p[p2][1]),int(pc.p[p2][0])]
                cv2.line(dbg, a, b, (0,255,255), 1, cv2.LINE_AA)
        
        cv2.imshow("test", dbg)
        cv2.waitKey(0)