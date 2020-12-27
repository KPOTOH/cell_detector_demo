"""
This class was made by Aditya M. Deshpande (https://github.com/adipandas/multi-object-tracker),
And was modified by Bogdan Efimenko (https://github.com/KPOTOH)
"""

from collections import OrderedDict
import sys

from scipy.spatial import distance
import numpy as np

print('You use ultra mega centroid multitracker', file=sys.stderr)

class centroid_multi_tracker:
    def __init__(self, maxLost=4, max_jump=80, size=(1114, 832)):           # maxLost: maximum object lost counted when the object is being tracked
        self.nextObjectID = 0                   # ID of next object
        self.objects = OrderedDict()
        self.size = size
        self.max_jump = max_jump
        self.maxLost = maxLost                  # maximum number of frames object was not detected.
        
    def addObject(self, centroid, box, mask, score=.71,):
        self.objects[self.nextObjectID] = dict(
            centroid=centroid,
            box=box,
            lost=0,
            detected=True,
            score=score,
            mask=mask
        )
        self.nextObjectID += 1
    
    def removeObject(self, objectID):                          # remove tracker data after object is lost
        del self.objects[objectID]
    
    @staticmethod
    def getLocation(bounding_box):
        xlt, ylt, xrb, yrb = bounding_box
        return ((xlt + xrb) / 2, (ylt + yrb) / 2)

    @staticmethod
    def my_argmin(D, max_jump):
        row_idx = D.min(axis=1).argsort()
        mask = (D < max_jump).astype(int)
        possible_num_of_min_col = mask.sum(axis=1)
        
        cols_idx = np.zeros_like(possible_num_of_min_col)
        cols_idx += max(D.shape) + 7
        used_col_idx = []

        for cid,rid in enumerate(row_idx):
            if possible_num_of_min_col[rid] == 1:
                for i in np.argsort(D[rid]):
                    if i not in used_col_idx and mask[rid, i] == True:
                        cols_idx[cid] = i
                        break
                used_col_idx.append(cols_idx[cid])

            elif possible_num_of_min_col[rid] > 1:
                for i in np.argsort(D[rid]):
                    if i not in used_col_idx and mask[rid, i] == True:
                        cols_idx[cid] = i
                        break
                used_col_idx.append(cols_idx[cid])

            elif possible_num_of_min_col[rid] == 0:
                pass
            else:
                raise Exception('fuck you')
        
        bad_idx = np.where(cols_idx == max(D.shape) + 7)[0]
        row_idx, cols_idx = list(row_idx), list(cols_idx)

        for bid in bad_idx[::-1]:
            del row_idx[bid]
            del cols_idx[bid]
        
        return row_idx, cols_idx
    

    def update(self, detections, masks, scores):
        
        if len(detections) == 0:   # if no object detected in the frame
            for objectID in self.objects.keys():
                self.objects[objectID]['lost'] += 1
                if self.objects[objectID]['lost'] > self.maxLost:
                    self.removeObject(objectID)
            
            return self.objects
        
        new_object_centroids = np.zeros((len(detections), 2))     # current object locations

        for (i, detection) in enumerate(detections): 
            new_object_centroids[i] = self.getLocation(detection)
        
        if len(self.objects)==0:
            for i in range(0, len(detections)): 
                self.addObject(new_object_centroids[i], detections[i], masks[i], scores[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            
            pr_real_centroids = previous_object_centroids * self.size
            new_real_centroids = new_object_centroids * self.size

            D = distance.cdist(pr_real_centroids, new_real_centroids) # pairwise distance between previous and current
 
            row_idx,cols_idx = self.my_argmin(D, self.max_jump)

            assignedRows, assignedCols = set(), set()
            
            for (row, col) in zip(row_idx, cols_idx):
                
                if row in assignedRows or col in assignedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = dict(
                    centroid=new_object_centroids[col],
                    box=detections[col],
                    lost=0,
                    detected=True,
                    score=scores[col],
                    mask=masks[col]
                )
                
                assignedRows.add(row)
                assignedCols.add(col)
                
            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)
            
            
            for row in unassignedRows:
                objectID = objectIDs[row]
                self.objects[objectID]['lost'] += 1
                self.objects[objectID]['detected'] = False
                self.objects[objectID]['score'] = .71
                # self.objects[objectID]['mask'] = None
                
                if self.objects[objectID]['lost'] > self.maxLost:
                    self.removeObject(objectID)
                        
            for col in unassignedCols:
                self.addObject(new_object_centroids[col], detections[col], masks[col], scores[col])
            
        return self.objects