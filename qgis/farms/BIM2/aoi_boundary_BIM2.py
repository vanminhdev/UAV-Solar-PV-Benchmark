import os
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsVectorFileWriter

def create_padded_aoi():
    project = QgsProject.instance()
    layers = project.mapLayersByName("modules")
    
    if not layers:
        print("❌ Lỗi: Không tìm thấy lớp 'modules'!")
        return
        
    modules_layer = layers[0]
    
    geoms = [f.geometry() for f in modules_layer.getFeatures() if f.geometry() and not f.geometry().isEmpty()]
    union_geom = QgsGeometry.unaryUnion(geoms)
    
    # 1. Nở to 20m để vượt qua các con đường rộng ở BIM2
    dilated = union_geom.buffer(20.0, 8)
    
    # 2. Thu lại 19m để giữ lại đúng Padding 1m
    closed = dilated.buffer(-19.0, 8)
    
    # 3. Giữ lại đường bao ngoài của TẤT CẢ các khối, không vứt bỏ khối nào
    if closed.isMultipart():
        solid_parts = []
        for poly in closed.asMultiPolygon():
            exterior_ring = [poly[0]]
            solid_parts.append(QgsGeometry.fromPolygonXY(exterior_ring))
        solid_geom = QgsGeometry.collectGeometry(solid_parts)
    else:
        poly = closed.asPolygon()
        exterior_ring = [poly[0]]
        solid_geom = QgsGeometry.fromPolygonXY(exterior_ring)
        
    aoi_layer = QgsVectorLayer(f"Polygon?crs={modules_layer.crs().authid()}", "aoi_boundary", "memory")
    aoi_prov = aoi_layer.dataProvider()
    
    # KHẮC PHỤC LỖI TẠI ĐÂY: Khởi tạo rỗng rồi mới setGeometry
    aoi_feat = QgsFeature()
    aoi_feat.setGeometry(solid_geom)
    aoi_prov.addFeatures([aoi_feat])
    aoi_layer.updateExtents()
    
    # BẮN AOI TRỰC TIẾP LÊN BẢN ĐỒ:
    QgsProject.instance().addMapLayer(aoi_layer)
    print("✅ Đã tạo lớp AOI và hiển thị thành công lên màn hình!")
create_padded_aoi()