import numpy as np
from skimage import measure
from joblib import delayed, Parallel


def load_x_model(filepath):
    vertices = []
    triangles = []
    transform_matrix = np.eye(4)

    with open(filepath, 'r') as file:
        lines = file.readlines()
        section = None

        for line in lines:
            line = line.strip()

            if line.startswith('FrameTransformMatrix '):
                section = 'FrameTransformMatrix'
                continue
            elif line.startswith('Mesh '):
                section = 'Mesh'
                continue
            elif line.startswith('Triangles '):
                section = 'Triangles'
                continue

            if section:
                if section == 'FrameTransformMatrix':
                    try:
                        transform_matrix = parse_transform_matrix(line)
                    except ValueError:
                        pass
                elif line.startswith(';;'):
                    section = None
                elif line.endswith(';,'):
                    if section == 'Mesh':
                        parts = line[:-2].split(';')
                        if len(parts) == 3:
                            x, y, z = map(float, parts)
                            vertex = np.array([x, y, z, 1.0])
                            transformed_vertex = np.dot(transform_matrix, vertex)[:3]
                            vertices.append(transformed_vertex)
                    elif section == 'Triangles':
                        parts = line[2:-2].split(',')
                        if len(parts) == 3:
                            x, y, z = map(int, parts)
                            triangles.append((x, y, z))

    return np.array(vertices), np.array(triangles)


def parse_transform_matrix(line):
    parts = line.split(';')[0].split(',')
    numbers = [float(part) for part in parts]
    matrix = np.array(numbers).reshape((4, 4))
    return matrix.T


def process_triangle(triangles, vertices, min_bounds, scale, grid_size):
    local_voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]

        v0_scaled = np.clip(((v0 - min_bounds) * scale).astype(int), 0, grid_size - 1)
        v1_scaled = np.clip(((v1 - min_bounds) * scale).astype(int), 0, grid_size - 1)
        v2_scaled = np.clip(((v2 - min_bounds) * scale).astype(int), 0, grid_size - 1)

        x_coords = [v0_scaled[0], v1_scaled[0], v2_scaled[0]]
        y_coords = [v0_scaled[1], v1_scaled[1], v2_scaled[1]]
        z_coords = [v0_scaled[2], v1_scaled[2], v2_scaled[2]]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    pt = np.array([x, y, z])
                    b1 = np.cross(v1_scaled - v0_scaled, pt - v0_scaled)[2] >= 0
                    b2 = np.cross(v2_scaled - v1_scaled, pt - v1_scaled)[2] >= 0
                    b3 = np.cross(v0_scaled - v2_scaled, pt - v2_scaled)[2] >= 0

                    if b1 == b2 == b3:
                        local_voxel_grid[x, y, z] = True

    return local_voxel_grid


def voxelization(vertices, triangles, grid_size=256):
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    scale = (grid_size - 1) / (max_bounds - min_bounds)
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    num_chunks = 8
    chunks = np.array_split(triangles, num_chunks)
    results = Parallel(n_jobs=num_chunks)(delayed(process_triangle)(chunk, vertices, min_bounds, scale, grid_size)
                                          for chunk in chunks)

    for result in results:
        voxel_grid |= result

    return voxel_grid


def create_mesh_from_voxels(voxel_grid):
    vertices, faces, _, _ = measure.marching_cubes(voxel_grid, 0)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    return vertices, faces


def save_model(vertices, faces, filepath, scale_factor=1.9):
    scaled_vertices = vertices.copy()
    scaled_vertices[:, 0] *= scale_factor
    lines = [f"v {v[0]} {v[1]} {v[2]}\n" for v in scaled_vertices]
    lines += [f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n" for face in faces]
    with open(filepath, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    vertices1, triangles1 = load_x_model('inputs/teapot_1.x')
    vertices2, triangles2 = load_x_model('inputs/teapot_2.x')

    vertices = np.concatenate((vertices1, vertices2))
    triangles = np.concatenate((triangles1, triangles2 + len(vertices1)))

    voxel_grid = voxelization(vertices, triangles)

    vertices, faces = create_mesh_from_voxels(voxel_grid)

    save_model(vertices, faces, 'results/combined_teapot.obj')
