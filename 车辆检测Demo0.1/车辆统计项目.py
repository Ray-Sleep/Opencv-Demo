# 加载视频
import cv2
import numpy as np

cap = cv2.VideoCapture('./video.mp4')
bgs = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

"""
    min_w
"""
min_w, max_w = 90, 160
min_h, max_h = 80, 150

line_high = 500
# 偏移量
offset = 7

cars = []
carNum = 0

# 计算外接矩形的中心点
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = int(x) + x1
    cy = int(y) + y1
    return cx, cy


# 循环读取视频
while True:
    ret, frame = cap.read()
    if ret:  # ret 为Boolean值
        # 把原始帧进行灰度化，然后去噪
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯去噪
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        fgmask = bgs.apply(blur)

        # 腐蚀
        erode = cv2.erode(fgmask, kernel)
        # 膨胀，还原图像大小
        dilate = cv2.dilate(erode, kernel, iterations=2)  # 二次膨胀加大效果

        # 消除内部的小块
        # 闭运算
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓 返回轮廓和层级
        contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 画出检测线
        cv2.line(frame, (20, line_high), (1250, line_high), (255, 255, 0), 3)

        # 画出所有检测出来的轮廓
        for contour in contours:
            # 最大外接矩形
            (x, y, w, h) = cv2.boundingRect(contour)

            # 通过外接矩形的宽高大小来过滤掉大矩形中的小矩形
            is_valid = (min_w <= w) & (min_h <= h)
            if not is_valid:
                continue

            # 要求坐标点都是整数
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 将车抽象为一点，即外接矩形的中心点
            # 通过外接矩形计算矩形的中心点
            cpoint =   center(x, y, w, h)
            cars.append(cpoint)
            cv2.circle(frame,(cpoint),5,(0,0,255),-1)

            # 判断汽车是否过检验线
            for (x,y) in cars:
                if y > (line_high - offset) and y < (line_high + offset):
                    # 进入有效区间
                    carNum += 1
                    cars.remove((x,y))
                    print(carNum)
        # 划线统计（画线部分）
        cv2.putText(frame,"Vehicle Count:"+str(carNum),(500,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
        cv2.imshow('frame', frame)

    key = cv2.waitKey(1)  # 多少毫秒每帧
    # 用户按esc退出
    if key == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
