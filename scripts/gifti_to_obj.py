import numpy as np


def export_to_obj(vertices, triangles, output_path):
    obj_str = ''
    for i in range(len(vertices)):
        obj_str += 'v ' + ' '.join(vertices[i].astype(str)) + '\n'
    obj_str += '\n'
    for i in range(len(triangles)):
        obj_str += 'f ' + ' '.join((triangles[i] + 1).astype(str)) + '\n'
    with open(output_path, 'w') as f:
        f.write(obj_str)

def export_regions():
    for label in np.unique(labels):
        matching_vertices = vertices[labels == label]
        matching_triangles = triangles[np.all(labels[triangles] == label, axis=1)]
        index_mapping = dict() # re-indexing map for the region
        for i, j in enumerate(np.argwhere(labels == label).flatten()):
            index_mapping[j] = i
        export_to_obj(
            vertices=matching_vertices, 
            triangles=np.vectorize(index_mapping.get)(matching_triangles), 
            output_path='/tmp/regions/{}.obj'.format(label_table[label])
        )
