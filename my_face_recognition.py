import face_recognition
from PIL import Image, ImageDraw


def face_rec():
    face_img = face_recognition.load_image_file('img/1.jpg')
    face_locations = face_recognition.face_locations(face_img)

    print(face_locations)
    # print(f'Лиц на фото - {len(face_locations)}')

    pil_image = Image.fromarray(face_img)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left) in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw

    pil_image.save('img/new.jpg')


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for face_location in faces_locations:
        top, right, bottom, left = face_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f'img/{count}_face_img.jpg')
        count += 1
    return f'Количество лиц на фото - {count}'


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    # print(img1_encodings)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    # print(result)

    if result[0]:
        print('Один человек')
    else:
        print('Разные люди')


def main():
    face_rec()
    print(extracting_faces('img/1.jpg'))
    compare_faces('img/2.jpg', 'img/3.jpg')
    compare_faces('img/3.jpg', 'img/4.jpg')


if __name__ == '__main__':
    main()
