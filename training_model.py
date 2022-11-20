import os
import pickle
import sys
import face_recognition


def train_model_by_img(name):

    if not os.path.exists('dataset'):
        print('Ошибка. Нет директории dataset')
        sys.exit()

    known_encodings = []
    images = os.listdir('dataset')

    print(images)

    for (i, image) in enumerate(images, 1):
        print(f'[+] processing img {i}/{len(images)}')
        print(image)

        face_img = face_recognition.load_image_file(f'dataset/{image}')
        face_locations = face_recognition.face_locations(face_img)
        if face_locations:
            face_enc = face_recognition.face_encodings(face_img)[0]
        else:
            print('Лицо не обнаружено')
            continue

        # print(face_enc)

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])
                print(result)

                if result[0]:
                    known_encodings.append(face_enc)
                    print('Same person!')
                    break
                else:
                    print('Another person!')
                    break

    print(known_encodings)
    print(f'длина {len(known_encodings)}')

    data = {
        'name': name,
        'encodings': known_encodings
    }

    with open(f'{name}_encoding.pickle', 'wb') as file:
        file.write(pickle.dumps(data))

    return f'[INFO] File {name}_encoding.pickle successfully created'

def main():
    print(train_model_by_img('name'))


if __name__ == '__main__':
    main()
