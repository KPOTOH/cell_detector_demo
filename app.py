import gc
import tempfile

import cv2
from PIL import Image
import numpy as np
import streamlit as st

from utils.predictor import (
    CellDetector, get_track, write_video, plot_trajectories_plot
)

DEFAULT_VIDEO_FILEPATHES = [
    'data/videos/Well_A1_05.webm',
    'data/videos/Well_A1_08.webm',
    'data/videos/Well_A2_25.webm',
]

FAQ = """
Вам представляется демонстрация детектора одного из типов стволовых
клеток, с которым часто работают в медицине, в том числе регенеративной.

Существует несколько задач, связанных с этими клетками, визуализированными
с помощью фазово-контрастной микроскопии. Все сводится к подсчитыванию
количества клеток, их размеров и направлений миграций. В среду, в которой
ползают клетки, может быть добавлен некий материал, который может либо
нравиться клеткам, либо нет, тогда они будут от него уползать или
дохнуть (подсчетом трупов я пока не занимался).

Задача популярная, а софта нормального в открытом доступе нет, поэтому
клетки считают люди, на что у них уходят месяцы, ведь количество данных
с экспериментов колоссально, а количество экспериментов тоже велико.
В связи с этим, было решено что-то с этим сделать. И было сделано.

Разметили данные. Обучили **Mask R-CNN R-50**. И получили красивые видеозаписи.
Данная демонстрация представляет собой суррогат решения, она внутри не дергает
нейросетку, этот момент будет решен в будущем.

Предсказания были сделаны для одной видеозаписи (на самом деле
для многих, но покажу только одну), которая показывает события в
одной лунке на планшете со средой в рамках одного эксперимента в течение
1-2 недель.

Однако, правила хорошего тона соблюдены: эмулятор предиктора,
который обращается к записям в файлах, написан, а трекер клеток обновляется
в реальном времени, обрабатывая некогда сделанные предсказания.

Под предсказаниями приведена минимальная статистика, которую в больших
количествах используют в дальнейшем анализе данных.

Чуть более подробное описание решения и [код](https://github.com/kpotoh/cell-segm-tracker)

[Материалы](https://drive.google.com/drive/folders/1avI-stcBHE_uibT2QkPzf3DC9LMhTO0f?usp=sharing),
которые можно загрузить на данной странице и ничего не сломается
"""

detector = CellDetector()


def get_filepath_of_loaded_file(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    filepath = tfile.name
    return filepath


def process_image():
    st.write("""## You can load image or select pictures from example """)
    option = st.radio(
        '',
        [f'image {i + 1}' for i in range(10)] + ['load image'],
    )
    image = None
    if option == 'load image':
        uploaded_file = st.file_uploader(
            "Choose another file", type=['png', 'jpg'])
        if uploaded_file is not None:
            image_path = uploaded_file.name
            image = Image.open(uploaded_file)
        else:
            st.warning('Load image')
            st.stop()
    else:
        idx = int(option[-1]) - 1 if int(option[-1]) != 0 else 9
        image_path = f'data/images/other_{idx}.jpg'
        image = Image.open(image_path)

    try:
        if image:
            image = np.asarray(image)
            st.header("Source image")
            st.image(image, use_column_width=True)

            st.header("Predictions")
            idx = int(image_path.split('.')[-2][-1])
            image_prediction = detector.predict_image(image, idx)
            predicted_image = image_prediction[0]
            st.image(predicted_image, use_column_width=True)

            st.header("Simple statistics")
            boxes = image_prediction[1]
            masks = image_prediction[3]
            ncells = len(boxes)
            cell_square_mean = masks.sum(axis=(1, 2)).mean()
            cell_square_std = masks.sum(axis=(1, 2)).std()
            st.markdown(
                f"""
                Detected **{ncells}** cells;

                Average cell square = **{cell_square_mean:.1f} with
                $\sigma$ = {cell_square_std:.1f}** $pix^2$ ;
                """
            )
        else:
            st.warning('Load image')
    except BaseException:
        st.warning('Load another image or refresh page')
        st.stop()


def process_video():
    st.write("""## You can load video or select default one """)
    option = st.radio(
        '',
        [f'video {i + 1}' for i in range(3)] + ['load video'],
    )
    if option == 'load video':
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['mp4', 'avi', 'webm'],
        )
        if uploaded_file is None:
            st.stop()
        video_path = get_filepath_of_loaded_file(uploaded_file)

    else:
        video_idx = int(option[-1]) - 1
        video_path = DEFAULT_VIDEO_FILEPATHES[video_idx]

    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.header("Source video")
        st.video(video_bytes)
    
    st.markdown('### Cells are crawling the medium on plate. We must count them...')
    if not st.button("Start prediction!"):
        st.stop()
    try:
        st.text('Patience you must have')
        video_prediction = detector.predict_video(video_path)
        predicted_video_path = write_video(video_prediction)
        tracks = get_track(video_prediction)
    except BaseException:
        st.warning('Load another image or refresh page')
        st.stop()

    with open(predicted_video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.header("Predicted video")
        st.video(video_bytes)

    st.header('Simple statistics')
    ncells = len(tracks)
    _cell_square_means = []
    _cell_square_stds = []
    for pred in video_prediction:
        if len(pred[1]) > 0:
            masks = pred[3]
            _cell_square_means.append(masks.sum(axis=(1, 2)).mean())
            _cell_square_stds.append(masks.sum(axis=(1, 2)).std())

    cell_square_mean = np.mean(_cell_square_means)
    cell_square_std = np.mean(_cell_square_stds)
    st.markdown(
        f"""
        Detected **{ncells}** cells;

        Average cell square = **{cell_square_mean:.1f} with
        $\sigma$ = {cell_square_std:.1f}** $pix^2$ ;
        """
    )
    fig = plot_trajectories_plot(tracks)
    st.pyplot(fig)
    expander = st.beta_expander("Centered plot FAQ")
    expander.markdown("""
    Трек от каждой клетки пускаем из (0, 0) и смотрим, куда направлен
    тренд движения клеток в среднем
    """)


def main():
    st.title("Cell detector on video or images")
    expander = st.beta_expander("FAQ")
    expander.markdown(FAQ)

    option = st.radio(
        'What do you like to process?',
        ('Image', 'Video'),
    )

    if option == 'Image':
        process_image()
        gc.collect()
    elif option == 'Video':
        process_video()
        gc.collect()


if __name__ == "__main__":
    main()
