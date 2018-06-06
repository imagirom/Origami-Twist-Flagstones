import svgwrite
import os
import numpy as np


def save_cp(filename, polygons, connections=None, centers=None, center_connections=None):
    lines = []
    if connections is not None:
        for con in connections:
            lines.append(np.array([polygons[con[0, 0]][con[0, 1]], polygons[con[1, 0]][con[1, 1]]]))

    if centers is not None:
        for i, center in enumerate(centers):
            for con in center_connections[i]:
                lines.append(np.stack([polygons[con[0]][con[1]], center]))
                lines.append(np.stack([polygons[con[0]][(con[1] - 1) % len(polygons[con[0]])], center]))

    for poly in polygons:
        lines.extend(np.stack([poly, np.concatenate([poly[1:], poly[[0]]])], axis=1))

    path = os.path.join('saved_CPs', filename)
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    svg_document = svgwrite.Drawing(filename=path,
                                    size=("80px", "60px"))

    lines = np.array(lines)
    lenghts = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=-1)
    width = np.mean(lenghts)/3
    print(f'saving {len(lines)} lines')
    for i, line in enumerate(lines):
        line = line.astype(float)
        svg_document.add(svgwrite.shapes.Line(start=line[0], end=line[1],
                                              stroke_width=str(width / 20),
                                              stroke="black", ))
    svg_document.save()
