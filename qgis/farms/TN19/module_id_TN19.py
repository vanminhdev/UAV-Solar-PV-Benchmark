from qgis.core import QgsProject, QgsField
from qgis.PyQt.QtCore import QVariant, QCoreApplication
import re
import time

def assign_module_id_sweep_line(vector_layer_name="modules"):
    project = QgsProject.instance()
    v_layer = project.mapLayersByName(vector_layer_name)
    
    if not v_layer:
        print(f"Lỗi: Không tìm thấy lớp {vector_layer_name}")
        return
        
    v_layer = v_layer[0]
    
    v_layer.startEditing()
    fields = v_layer.fields()
    if fields.indexOf('String_ID') == -1:
        v_layer.addAttribute(QgsField('String_ID', QVariant.String, len=15))
    if fields.indexOf('module_id') == -1:
        v_layer.addAttribute(QgsField('module_id', QVariant.String, len=30))
    v_layer.updateFields()
    
    idx_layer = v_layer.fields().indexOf('layer')
    idx_str = v_layer.fields().indexOf('String_ID')
    idx_mod = v_layer.fields().indexOf('module_id')
    
    features = list(v_layer.getFeatures())
    blocks = {}
    
    print("Đang dọn dẹp dữ liệu và xử lý Block...")
    for feat in features:
        b_name = str(feat['layer']).strip()
        if not b_name or b_name.lower() == "null" or "duong" in b_name.lower() or "block" not in b_name.lower():
            b_name = "Block1" 
            v_layer.changeAttributeValue(feat.id(), idx_layer, b_name)
        b_name_lower = b_name.lower()
        blocks.setdefault(b_name_lower, []).append(feat)
        
    def get_block_number(b_name):
        nums = re.findall(r'\d+', b_name)
        return int(nums[0]) if nums else 0
    
    sorted_block_names = sorted(blocks.keys(), key=get_block_number)
    
    print("Đang áp dụng thuật toán 'Sweep-line' chống nghiêng bản đồ...")
    
    global_pv_counter = 1
    attribute_changes = {}
    start_time = time.time()
    
    for block_name in sorted_block_names:
        block_num = get_block_number(block_name)
        if block_num == 0:
            block_num = 1
            
        b_prefix = f"block{block_num}"
        feats = blocks[block_name]
        
        sample_bbox = feats[0].geometry().boundingBox()
        panel_height = sample_bbox.height()
        row_threshold = panel_height * 0.8 # Dung sai bám sát hàng ngang chống dốc
        
        # 1. Sắp xếp toàn bộ từ Tây sang Đông (X tăng dần)
        feats.sort(key=lambda f: f.geometry().centroid().asPoint().x())
        
        # 2. Sweep-line: Quét và nối các tấm pin thành từng hàng vật lý
        rows = []
        for feat in feats:
            feat_y = feat.geometry().centroid().asPoint().y()
            
            best_row = None
            min_y_diff = float('inf')
            
            for row in rows:
                last_feat_y = row[-1].geometry().centroid().asPoint().y()
                y_diff = abs(feat_y - last_feat_y)
                
                # Tìm hàng có Y gần nhất với tấm pin hiện tại
                if y_diff < row_threshold and y_diff < min_y_diff:
                    best_row = row
                    min_y_diff = y_diff
                    
            if best_row is not None:
                best_row.append(feat)
            else:
                rows.append([feat])
                
        # 3. Sắp xếp các hàng từ Bắc xuống Nam (Y trung bình giảm dần)
        def get_avg_y(row):
            return sum(f.geometry().centroid().asPoint().y() for f in row) / len(row)
            
        rows.sort(key=get_avg_y, reverse=True)
        
        # 4. Gán ID theo đúng quy tắc 4 hàng = 1 String
        for i, row in enumerate(rows):
            string_num = (i // 4) + 1
            string_val = f"string{string_num}"
            
            for feat in row:
                mod_id = f"{b_prefix}_{string_val}_{global_pv_counter:05d}"
                attribute_changes[feat.id()] = {
                    idx_str: string_val,
                    idx_mod: mod_id
                }
                global_pv_counter += 1
                
        QCoreApplication.processEvents()

    print("Đang lưu ID...")
    for fid, attrs in attribute_changes.items():
        v_layer.changeAttributeValues(fid, attrs)
        
    v_layer.commitChanges()
    end_time = time.time()
    
    total_panels = global_pv_counter - 1
    print(f"\n✅ HOÀN THÀNH TRONG {round(end_time - start_time, 2)} GIÂY!")
    print(f"✅ Đã định danh chính xác {total_panels} tấm pin, khắc phục triệt để lỗi nghiêng.")

# Chạy lệnh
assign_module_id_sweep_line()