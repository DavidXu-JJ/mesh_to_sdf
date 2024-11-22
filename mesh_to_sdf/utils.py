import functools
import trimesh
import numpy as np

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def forward_scale_to_unit_cube(mesh):
    # If the input is a Scene, merge it into a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    # Compute the translation and scale factors
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents
    scale = 2 / np.max(extents)

    # Translate and scale vertices
    vertices = mesh.vertices - centroid
    vertices *= scale

    # Return the transformed mesh and the inverse transformation parameters
    transformed_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    inverse_params = {
        'centroid': centroid,
        'scale': scale,
    }
    return transformed_mesh, inverse_params

def reverse_vertices(original_vertices, inverse_params):
    vertices = original_vertices / scale
    vertices += centroid
    return vertices

def reverse_scale_to_unit_cube(mesh, inverse_params):
    # Extract the inverse transformation parameters
    centroid = inverse_params['centroid']
    scale = inverse_params['scale']

    # Reverse the scaling and translation
    vertices = reverse_vertices(mesh.vertices, inverse_params)

    # Return the restored mesh
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

# Use get_raster_points.cache_clear() to clear the cache
@functools.lru_cache(maxsize=4)
def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points

def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
