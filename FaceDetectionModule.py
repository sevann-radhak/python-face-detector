import cv2
import mediapipe as mp
import time


class FaceDetector:    
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpDetection.FaceDetection(self.minDetectionCon)


    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
                   
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    # cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        return img, bboxs
    
    
    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        color = (255, 0, 255)
        
        # Top Left x, y
        cv2.line(img, (x, y), (x+l, y), color, t)
        cv2.line(img, (x, y), (x, y+l), color, t)
        
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1-l, y), color, t)
        cv2.line(img, (x1, y), (x1, y+l), color, t)
        
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x+l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1-l), color, t)
        
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1-l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1-l), color, t)
        
        return img


def main():
    # cap = cv2.VideoCapture('videos/2.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)   
        pTime = cTime
    
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imshow('Video', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()