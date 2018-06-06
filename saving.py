import svgwrite
import os

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
    if not os.path.exists(path):
        os.mkdir(os.path.dirname(path))

    svg_document = svgwrite.Drawing(filename=path,
                                    size=("8000px", "6000px"))

    lines = np.array(lines)
    print(np.min(np.linalg.norm(lines.flatten(), axis=-1)))
    _, idx = np.unique(np.sort(np.sort(lines, axis=1), axis=2) // 100, axis=0, return_index=True)
    idx = np.array(idx, dtype=np.int32)
    print(idx.shape)
    for line in lines[idx.flatten()]:
        line = line.astype(float)
        svg_document.add(svgwrite.shapes.Line(start=line[0], end=line[1],
                                              stroke_width="10",
                                              stroke="black", ))
    svg_document.save()
