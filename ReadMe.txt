<Mask RCNN 2kcal>
Mask RCNN을 통해 Mask를 구하고 음식(현재는 과일 3종류: 사과, 바나나, 오렌지)의 칼로리를 측정하는 프로그램

1. requirements에 있는 모듈을 모두 다운.(pip install -r requirements.txt) 
(pycocotools문제가 생길 경우 추가로 pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI")
2. https://github.com/matterport/Mask_RCNN/releases 에서 Mask R-CNN 2.0의 mask_rcnn_coco.h5 다운 받아 mrcnn with coco_no dataset.py가 있는 디렉토리에 저장
3. images 폴더에서 칼로리를 구하고자 하는 사진 삽입.(다수를 넣을 경우 랜덤하게 선택)
4. mrcnn with coco_no dataset.py 실행하여 칼로리 계산.
