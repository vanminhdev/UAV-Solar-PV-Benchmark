import math
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, 
                       QgsPointXY, QgsField, QgsWkbTypes,
                       QgsFillSymbol, QgsSingleSymbolRenderer) # <-- Thêm thư viện xử lý đồ họa
from qgis.utils import iface
from PyQt5.QtCore import QVariant

# --- CẤU HÌNH KÍCH THƯỚC CHUẨN ---
PANEL_WIDTH = 1.029   # Bề ngang chuẩn
PANEL_HEIGHT = 1.982  # Chiều dọc của 1 tấm pin
NUM_ROWS = 4          # Số hàng pin

layer = iface.activeLayer()

if not layer or layer.geometryType() != QgsWkbTypes.LineGeometry:
    print("LỖI: Vui lòng click chọn lớp 'Duong_Chuan' ở bảng Các Lớp trước khi chạy!")
else:
    crs = layer.crs().authid()
    poly_layer = QgsVectorLayer(f"Polygon?crs={crs}", "Luoi_TuDong", "memory")
    pr = poly_layer.dataProvider()
    
    pr.addAttributes([QgsField("ID_Pin", QVariant.Int),
                      QgsField("Row_ID", QVariant.Int),
                      QgsField("Tam_X", QVariant.Double),
                      QgsField("Tam_Y", QVariant.Double)])
    poly_layer.updateFields()

    features_to_add = []
    panel_id = 1
    
    for feat in layer.getFeatures():
        line = feat.geometry().asPolyline()
        if len(line) < 2: continue
            
        p1, p2 = line[0], line[-1]
        
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        angle = math.atan2(dy, dx)
        total_length = math.sqrt(dx**2 + dy**2)
        
        num_cols = round(total_length / PANEL_WIDTH)
        if num_cols < 1: num_cols = 1
        
        for row in range(NUM_ROWS):
            for col in range(num_cols):
                x_base = p1.x() + col * PANEL_WIDTH * math.cos(angle) - row * PANEL_HEIGHT * math.sin(angle)
                y_base = p1.y() + col * PANEL_WIDTH * math.sin(angle) + row * PANEL_HEIGHT * math.cos(angle)
                
                pt1 = QgsPointXY(x_base, y_base)
                pt2 = QgsPointXY(x_base + PANEL_WIDTH * math.cos(angle), y_base + PANEL_WIDTH * math.sin(angle))
                pt3 = QgsPointXY(pt2.x() - PANEL_HEIGHT * math.sin(angle), pt2.y() + PANEL_HEIGHT * math.cos(angle))
                pt4 = QgsPointXY(pt1.x() - PANEL_HEIGHT * math.sin(angle), pt1.y() + PANEL_HEIGHT * math.cos(angle))
                
                poly = QgsGeometry.fromPolygonXY([[pt1, pt2, pt3, pt4, pt1]])
                centroid = poly.centroid().asPoint()
                
                f = QgsFeature()
                f.setGeometry(poly)
                f.setAttributes([panel_id, row + 1, centroid.x(), centroid.y()])
                features_to_add.append(f)
                panel_id += 1
            
    pr.addFeatures(features_to_add)
    
    # ==========================================
    # --- TỰ ĐỘNG HÓA HIỂN THỊ (STYLE) ---
    # ==========================================
    symbol = QgsFillSymbol.createSimple({
        'style': 'no',           # Kiểu tô: Không có vẽ (No Brush)
        'outline_color': 'red',  # Màu nét: Đỏ
        'outline_width': '0.26'  # Độ dày nét mặc định
    })
    poly_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    
    QgsProject.instance().addMapLayer(poly_layer)
    print(f"THÀNH CÔNG! Đã tạo tự động tổng cộng {panel_id - 1} ô lưới cho {NUM_ROWS} hàng.")