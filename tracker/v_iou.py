import numpy as np
from lapsolver import solve_dense
from util import iou
from tracker.vis_tracker import VisTracker

class V_IOU:
    def __init__(self, device, tracker_type, sigma_l, sigma_h, sigma_iou, t_min, ttl, keep_upper_height_ratio) -> None:
        self.tracks_active = []
        self.tracks_extendable = []
        self.tracks_finished = []
        self.frame_buffer = []
        self.max_id = 0

        self.device = device
        self.tracker_type = tracker_type
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.ttl = ttl 
        self.keep_upper_height_ratio = keep_upper_height_ratio

    def update(self, frame_idx, frame, detections):
        self.height, self.width = frame.shape[:2]
        
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.ttl:
            self.frame_buffer.pop(0)
        
        detections = [det for det in detections if det['score'] >= self.sigma_l]
        tracks_id, detections_id = self.associate(detections)

        updated_tracks = []

        for track_id, detection_id in zip(tracks_id, detections_id):
            self.tracks_active[track_id]['bboxes'].append(detections[detection_id]['bbox'])
            self.tracks_active[track_id]['max_score'] = max(self.tracks_active[track_id]['max_score'], detections[detection_id]['score'])
            self.tracks_active[track_id]['classes'].append(detections[detection_id]['class'])
            self.tracks_active[track_id]['det_counter'] += 1

            if self.tracks_active[track_id]['ttl'] != self.ttl:
                # reset visual tracker if active
                self.tracks_active[track_id]['ttl'] = self.ttl
                self.tracks_active[track_id]['visual_tracker'] = None
            
            updated_tracks.append(self.tracks_active[track_id])
        
        tracks_not_updated = [self.tracks_active[idx] for idx in set(range(len(self.tracks_active))).difference(set(tracks_id))]

        for track in tracks_not_updated:
            if track['ttl'] > 0 and frame_idx > 2:
                if track['ttl'] == self.ttl:
                    # init visual tracker
                    track['visual_tracker'] = VisTracker(self.tracker_type, track['bboxes'][-1], self.frame_buffer[-2],
                                                         self.keep_upper_height_ratio)
                # viou forward update
                ok, bbox = track['visual_tracker'].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    self.tracks_extendable.append(track)
                    continue

                track['ttl'] -= 1
                track['bboxes'].append(bbox)
                updated_tracks.append(track)
            else:
                self.tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        tracks_extendable_updated = []
        for track in self.tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + self.ttl - track['ttl'] >= frame_idx:
                tracks_extendable_updated.append(track)
            elif track['max_score'] >= self.sigma_h and track['det_counter'] >= self.t_min:
                self.tracks_finished.append(track)
        self.tracks_extendable = tracks_extendable_updated

        new_dets = [detections[idx] for idx in set(range(len(detections))).difference(set(detections_id))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(self.tracker_type, det['bbox'], frame, self.keep_upper_height_ratio)

            for f in reversed(self.frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(self.tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

                    offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_idx
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if 1 <= offset <= self.ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= self.sigma_iou:
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track['bboxes'] = track['bboxes'][:-offset+1]
                        track['bboxes'] += list(reversed(boxes))[1:]
                        track['bboxes'].append(det['bbox'])
                        track['max_score'] = max(track['max_score'], det['score'])
                        track['classes'].append(det['class'])
                        track['ttl'] = self.ttl
                        track['visual_tracker'] = None

                        self.tracks_extendable.remove(track)
                        if track in self.tracks_finished:
                            del self.tracks_finished[self.tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                det['id'] = self.max_id
                self.max_id += 1
                dets_for_new.append(det)

        # create new tracks
        new_tracks = [{'id': det['id'], 'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_idx, 'ttl': self.ttl,
                       'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det in dets_for_new]
        tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                self.tracks_extendable.append(track)
            else:
                tracks_active.append(track)
        
        self.tracks_active = tracks_active

        # finish all remaining active and extendable tracks
        self.tracks_finished = self.tracks_finished + \
                          [track for track in self.tracks_active + self.tracks_extendable
                           if track['max_score'] >= self.sigma_h and track['det_counter'] >= self.t_min]

         # remove last visually tracked frames and compute the track classes
        for track in self.tracks_finished:
            if self.ttl != track['ttl']:
                track['bboxes'] = track['bboxes'][:-(self.ttl - track['ttl'])]
            track['class'] = max(set(track['classes']), key=track['classes'].count)
        # filter out tracks with no bbox
        self.tracks_active = list( filter(lambda x: len(x['bboxes']) > 0, self.tracks_active) )
        return self.tracks_active

   

    #     del track['visual_tracker']

    # return tracks_finished

    def associate(self, detections):
        costs = np.empty(shape=(len(self.tracks_active), len(detections)), dtype=np.float32)

        for row, track in enumerate(self.tracks_active):
            for col, detection in enumerate(detections):
                costs[row, col] = 1 - iou(self._xywh_to_xyxy(track['bboxes'][-1]), self._xywh_to_xyxy(detection['bbox']))

        np.nan_to_num(costs)
        costs[costs > 1 - self.sigma_iou] = float('NaN')
        track_ids, det_ids = solve_dense(costs)
        
        return track_ids, det_ids
    
    def _xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2-x1, y2-y1]

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        return [x, y, x+w, y+h]

    def _xywh_to_tlwh(self, bbox):
        bbox_xywh = np.copy(bbox)
        if bbox_xywh is not None:
            bbox_tlwh = np.zeros(bbox_xywh.shape)
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()

        bbox_tlwh[0] = bbox_xywh[0] - bbox_xywh[2]/2.
        bbox_tlwh[1] = bbox_xywh[1] - bbox_xywh[3]/2. 
        bbox_tlwh[2] = bbox_xywh[2]
        bbox_tlwh[3] = bbox_xywh[3]  
        return bbox_tlwh


    # def _xywh_to_xyxy(self, bbox_xywh):
    #     x,y,w,h = bbox_xywh
    #     x1 = max(int(x-w/2),0)
    #     x2 = min(int(x+w/2),self.width-1)
    #     y1 = max(int(y-h/2),0)
    #     y2 = min(int(y+h/2),self.height-1)
    #     return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w), self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h), self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h

