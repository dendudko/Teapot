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


def point_to_triangle_distance(pt, v0, v1, v2):
    def project_point_on_line(a, b, p):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    def point_to_segment_distance(p, a, b):
        projection = project_point_on_line(a, b, p)
        return np.linalg.norm(p - projection)

    def point_to_plane_distance(p, a, b, c):
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        return np.dot(p - a, normal)

    d = point_to_plane_distance(pt, v0, v1, v2)
    sign = np.sign(d)
    d = abs(d)
    d = min(d, point_to_segment_distance(pt, v0, v1), point_to_segment_distance(pt, v1, v2),
            point_to_segment_distance(pt, v2, v0))
    return d * sign


def calculate_weight(v, viewpoint, normal, epsilon):
    vector_to_voxel = v - viewpoint
    distance_to_surface = np.linalg.norm(vector_to_voxel)
    weight = np.dot(normal, vector_to_voxel / distance_to_surface)

    if distance_to_surface > epsilon:
        return weight
    else:
        return weight * (distance_to_surface / epsilon)


def process_triangle_with_distances(triangles, vertices, min_bounds, scale, grid_size, viewpoint, epsilon):
    local_integral_grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)

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
                    pt = np.array([x, y, z], dtype=float)
                    pt_real = pt / scale + min_bounds
                    dist = point_to_triangle_distance(pt_real, v0, v1, v2)
                    weight = calculate_weight(pt_real, viewpoint, normal, epsilon)
                    local_integral_grid[x, y, z] += dist * weight

    return local_integral_grid


def voxelization_with_distances(vertices, triangles, grid_size=128, epsilon=1.0):
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    viewpoints = ((min_bounds + max_bounds) / 2, max_bounds)

    scale = (grid_size - 1) / (max_bounds - min_bounds)
    integral_grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    num_chunks = 5
    chunks = np.array_split(triangles, num_chunks)

    for i, viewpoint in enumerate(viewpoints):
        results = Parallel(n_jobs=num_chunks)(
            delayed(process_triangle_with_distances)(chunk, vertices, min_bounds, scale, grid_size, viewpoint, epsilon)
            for chunk in chunks
        )

        for result in results:
            integral_grid += result

    return integral_grid


def create_mesh_from_integral(voxel_grid, level=0.0):
    vertices, faces, _, _ = measure.marching_cubes(voxel_grid, level)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    return vertices, faces


def save_model(vertices, faces, filepath, scale_factor=1.7):
    scaled_vertices = vertices.copy()
    scaled_vertices[:, 0] *= scale_factor
    with open(filepath, 'w') as f:
        for v in scaled_vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


if __name__ == '__main__':
    for level in (0.0,):
        grid_size = 512

        vertices1, triangles1 = load_x_model('inputs/teapot_1.x')
        vertices2, triangles2 = load_x_model('inputs/teapot_2.x')

        vertices = np.concatenate((vertices1, vertices2))
        triangles = np.concatenate((triangles1, triangles2 + len(vertices1)))

        voxel_grid = voxelization_with_distances(vertices, triangles, grid_size=grid_size)
        vertices, faces = create_mesh_from_integral(voxel_grid, level)

        save_model(vertices, faces, 'results/combined_teapot_new.obj')
