from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
from qgis.utils import iface

# Lấy lớp đang được chọn
layer = iface.activeLayer()

# Ép màu viền đỏ, nền rỗng
symbol = QgsFillSymbol.createSimple({'style': 'no', 'outline_color': 'red', 'outline_width': '0.26'})
layer.setRenderer(QgsSingleSymbolRenderer(symbol))

# Cập nhật hiển thị ngay lập tức
layer.triggerRepaint()
iface.layerTreeView().refreshLayerSymbology(layer.id())