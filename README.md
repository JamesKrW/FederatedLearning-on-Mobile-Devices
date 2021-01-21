# 38prp
Federated Learning on Mobile Devices
#####文件功能说明：

| 文件/目录 名                 | 功能                                               |
| ---------------------------- | -------------------------------------------------- |
|data_process          |读取图片，处理成128维数组                             |
|model         |模型定义与训练过程|
|photo          |机器人上运行，检测人脸并识别|
|speak          |语音播报  |
|takephoto    |拍照，获取数据集                             |
|test              |用训练好的模型对单张照片进行检测|
|communication.py             |设备与服务器通信|
|train_client.py             |设备接收服务器参数进行训练，回传|
|train_ps.py             |服务器发送参数，进行权重更新|
