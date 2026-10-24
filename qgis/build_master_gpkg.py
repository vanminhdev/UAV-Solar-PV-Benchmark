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
    
    # 1. Nở to 6m để qua đường
    dilated = union_geom.buffer(6.0, 8)
    
    # 2. Thu lại 5m (padding 1m)
    closed = dilated.buffer(-5.0, 8)
    
    if closed.isMultipart():
        main_poly = max(closed.asMultiPolygon(), key=lambda poly: QgsGeometry.fromPolygonXY(poly).area())
        exterior_ring = [main_poly[0]]
        solid_geom = QgsGeometry.fromPolygonXY(exterior_ring)
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
    
    output_gpkg = os.path.join(project.homePath() or os.path.expanduser("~"), "thuannam19_master.gpkg")
    
    # KHẮC PHỤC LỖI UNPACK CHO LỚP AOI
    opts_aoi = QgsVectorFileWriter.SaveVectorOptions()
    opts_aoi.driverName = "GPKG"
    opts_aoi.layerName = "aoi_boundary"
    opts_aoi.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
    
    result_aoi = QgsVectorFileWriter.writeAsVectorFormatV3(
        aoi_layer, output_gpkg, project.transformContext(), opts_aoi
    )
    
    # KHẮC PHỤC LỖI UNPACK CHO LỚP MODULES
    opts_mod = QgsVectorFileWriter.SaveVectorOptions()
    opts_mod.driverName = "GPKG"
    opts_mod.layerName = "modules"
    opts_mod.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    
    result_mod = QgsVectorFileWriter.writeAsVectorFormatV3(
        modules_layer, output_gpkg, project.transformContext(), opts_mod
    )
    
    print("✅ Đã tạo AOI an toàn với Padding 1 mét!")

create_padded_aoi()