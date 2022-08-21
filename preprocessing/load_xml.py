import xml.dom.minidom


def load_xml(file_path):
   
    dom = xml.dom.minidom.parse(file_path)
    annotations = dom.getElementsByTagName('Annotation')

    xyi_in_annotations = []
    xyn_in_annotations = []

    for Annotation in annotations:

        xy_in_annotation = []
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


