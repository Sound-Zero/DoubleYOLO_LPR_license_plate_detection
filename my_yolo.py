from ultralytics import YOLO


def predict_once(model='',img_path='',conf=0.4,show=True,save=True,save_txt=True,detect_classes=[1,2,3,5,6,7]):

    if model:
        model=model
    else:
        model='./yolov8s.pt'#default_model

    if img_path:
        path=img_path
    else:
        path="./datasets/coco/images/120img"#default_img_path
    

    # detect_classes=[1,2,3,5,6,7]#限定检测的类别
    #   1: bicycle
    #   2: car
    #   3: motorcycle
    #   5: bus
    #   6: train
    #   7: truck
    yolo=YOLO(model=model)
    results=yolo.predict(source=path,imgsz=640,classes=detect_classes,conf=conf,show=show,save=save,save_txt=save_txt)    


def train_once(model='',data_yaml='',epochs=50+88,resume=False,batch=24):

    if model:
        model=model
    else:
        model="./yolov8s.pt"

    yolo=YOLO(model)
    yolo.load(model)
    print('模型加载完成')

    if data_yaml:
        data_yaml=data_yaml
    else:
        data_yaml=r"./YOLOdatasets/YOLO.yaml"


    yolo.train(
        data=data_yaml,
        imgsz=640,
        epochs=epochs,
        device=0,
        resume=resume,

        batch=batch,

    
    )

    print('运行完成')

if __name__ == '__main__':
    pass
