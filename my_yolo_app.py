# 导入所需的库
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os
import numpy as np

# 1. 设置网页的标题和布局
st.set_page_config(
    page_title="我的YOLO检测小助手",
    layout="wide"  # 使用宽屏布局，看起来更舒服
)
st.title("✨ 上传一张图片，看看YOLO发现了什么")

# 2. 在侧边栏放置控制按钮
with st.sidebar:
    st.header("⚙️ 控制面板")

    # 模型选择（如果你有多个模型，可以在这里扩展）
    model_option = st.selectbox(
        "选择模型",
        ["yolov8n.pt", "yolov8s.pt"]  # 你可以改成你的模型文件名
    )

    # 置信度阈值调节滑块
    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.25  # 默认值
    )

    st.divider()
    st.caption("🚀 由 Streamlit 和 YOLO 强力驱动")


# 3. 加载YOLO模型（使用@st.cache_resource实现缓存，避免每次操作都重新加载）
@st.cache_resource
def load_model(model_path):
    # 这里会加载你指定路径的模型
    return YOLO(model_path)


# 根据选择加载模型
model = load_model("yolov8n.pt")

# 4. 创建文件上传器
uploaded_file = st.file_uploader(
    "选择一张图片",
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG 或 PNG 格式"
)

# 5. 如果用户上传了文件，进行处理
if uploaded_file is not None:
    # 将上传的文件转换为OpenCV可以处理的格式
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8),
        cv2.IMREAD_COLOR
    )

    # 显示原始图片
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 原始图片")
        st.image(cv2_img, channels="BGR", use_container_width=True)

    # 进行目标检测
    with st.spinner("🔍 YOLO正在努力识别中..."):
        results = model(
            cv2_img,
            conf=confidence_threshold  # 使用用户设定的阈值
        )

    # 获取并显示标注后的图片
    annotated_img = results[0].plot()  # plot() 返回的是BGR格式的numpy数组

    with col2:
        st.subheader("🎯 检测结果")
        st.image(annotated_img, channels="BGR", use_container_width=True)

    # 显示检测到的物体清单
    if len(results[0].boxes) > 0:
        st.success(f"✅ 检测完成！共发现 {len(results[0].boxes)} 个目标")

        # 用表格展示检测详情
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "物体": class_name,
                "置信度": f"{confidence:.2f}"
            })
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("🤔 没有检测到任何目标，可以尝试调低置信度阈值。")

else:
    # 页面初始状态的提示
    st.info("👆 请从左侧上传一张图片开始体验。")
    # 这里可以放一些示例图片或说明