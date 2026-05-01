from ultralytics import YOLO


# 推理路径列表（一个个文件夹推理，保持与原版 inference.py 一致的使用方式）
# 需要测试哪个子数据集，就手动取消对应行的注释即可。
inference_source_list = [
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-动脉导管未闭/第1部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-动脉导管未闭/第2部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-房间隔缺损/第1部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-房间隔缺损/第2部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-房间隔缺损/第3部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-房间隔缺损/第4部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-房间隔缺损/第5部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第1部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第2部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第3部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第4部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第5部分",
    "/data/minghao_data/heart_disease_detect/heart_disease_detect-v3/test_new/test-v2-室间隔缺损/第6部分",
]


def inference(source):
    # 加载训练好的模型
    # 这里即使模型包含了辅助分割分支，predict() 仍然走标准 detection 输出，
    # save_txt / save_conf 的结果格式与原版 yolo26 检测脚本保持一致，可直接对接后处理脚本。
    model = YOLO("./runs/detect/train/weights/best.pt")

    # 推理
    inference_results = model.predict(
        source=source,
        imgsz=640,  # 与原脚本保持一致
        save=True,  # 与原脚本保持一致
        save_txt=True,  # 与原脚本保持一致
        save_conf=True,  # 与原脚本保持一致
        device=1,  # 与原脚本保持一致，按服务器实际情况修改
    )

    return inference_results


if __name__ == "__main__":
    for inference_source in inference_source_list:
        inference(inference_source)
