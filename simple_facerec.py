import face_recognition
import cv2
import os
import glob
import numpy as np

img_dizi = ["DiCaprio","Elon Musk","Jeff Bezos","Kivanc Tatlitug","Messi","Esra Bilgic","Tom Cruise"]


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Daha hızlı bir hız için çerçeveyi yeniden boyutlandırın
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # Resimleri Yükle
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Görüntü kodlamasını ve adlarını saklayın
        for img_path in images_path:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image: {img_path}")
                    continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Dosya adını yalnızca ilk dosya yolundan alın
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Kodlamayı al

            face_encodings = face_recognition.face_encodings(rgb_img)
            # Yüz tespit edilmediği durumu ele al
            if len(face_encodings) > 0:
                img_dizi = face_encodings[0]
            else:
                  continue

            # Dosya adını ve dosya kodlamasını saklayın
            self.known_face_encodings.append(img_dizi)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Videonun geçerli karesindeki tüm yüzleri ve yüz kodlamalarını bulun
        # Görüntüyü BGR renginden (OpenCV'nin kullandığı) RGB rengine (face_recognition'ın kullandığı) dönüştürün

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yüzün bilinen yüzlerle eşleşip eşleşmediğine bakın
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Bilinen_yüz_kodlamalarında bir eşleşme bulunursa, sadece ilkini kullanın. 
            # Eşleşmelerde True ise:
            # first_match_index = match.index(True) 
            # name = bilinen_yüz_adları[first_match_index]
            # Veya bunun yerine, yeni yüze en yakın mesafede olan bilinen yüzü kullanın
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Hızlı bir şekilde çerçeve yeniden boyutlandırma ile koordinatları ayarlamak için numpy dizisine dönüştürün

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names