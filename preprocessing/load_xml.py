import xml.dom.minidom


def load_xml(file_path):
    """读取xml文件，
    返回总坐标列表xy_list:存放一张图片上所有画出域的点 """

    # 用于打开一个xml文件，并将这个文件对象dom变量
    dom = xml.dom.minidom.parse(file_path)
    # 对于知道元素名字的子元素，可以使用getElementsByTagName方法获取
    annotations = dom.getElementsByTagName('Annotation')

    # 存放所有的 Annotation
    xyi_in_annotations = []
    xyn_in_annotations = []

    for Annotation in annotations:

        # 存放一个 Annotation 中所有的 X,Y值
        xy_in_annotation = []

        # 读取 Coordinates 下的 X Y 的值
        coordinates = Annotation.getElementsByTagName("Coordinate")
        for Coordinate in coordinates:
            list_in_annotation = []
            x = int(float(Coordinate.getAttribute("X")))
            y = int(float(Coordinate.getAttribute("Y")))

            list_in_annotation.append(x)
            list_in_annotation.append(y)

            xy_in_annotation.append(list_in_annotation)

        name_area = Annotation.getAttribute("Name")
        if name_area == "normal":
            xyn_in_annotations.append(xy_in_annotation)
        if name_area != "normal":
            xyi_in_annotations.append(xy_in_annotation)

    xy_tuple = (xyi_in_annotations, xyn_in_annotations)

    return xy_tuple


