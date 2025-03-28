import cv2, queue, threading
import time

class NoBufferVC:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        pass
      
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except queue.Empty:
          pass
      self.q.put(frame)

      time.sleep(1/(2 * self.fps))

  def read(self):
    return self.q.get()
  
  def get(self, property):
    return self.cap.get(property)
