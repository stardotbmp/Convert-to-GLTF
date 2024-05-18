from specklepy.objects.geometry import Mesh as SpeckleMesh
from specklepy.objects.other import Transform as SpeckleTransform

from collections.abc import Iterable
from typing import Optional, List, Tuple

from specklepy.objects import Base
from specklepy.objects.other import Transform, Instance

import trimesh
import numpy as np

import pyvista as pv


def combine_transform_matrices(transforms: list[SpeckleTransform]) -> np.ndarray:
    combined_matrix = np.identity(4)
    for transform in transforms:
        matrix = np.array(transform.value).reshape(4, 4)
        combined_matrix = np.dot(combined_matrix, matrix)
    return combined_matrix


def speckle_mesh_to_trimesh(speckle_mesh: SpeckleMesh) -> trimesh.Trimesh:
    # Convert Speckle mesh to trimesh format
    vertices = np.array(
        [
            speckle_mesh.vertices[i : i + 3]
            for i in range(0, len(speckle_mesh.vertices), 3)
        ]
    )
    faces = np.array(
        [speckle_mesh.faces[i : i + 3] for i in range(0, len(speckle_mesh.faces), 3)]
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces)


"""Helper module for a simple speckle object tree flattening."""


def flatten_base(base: Base) -> Iterable[Base]:
    """Flatten a base object into an iterable of bases.

    This function recursively traverses the `elements` or `@elements` attribute of the
    base object, yielding each nested base object.

    Args:
        base (Base): The base object to flatten.

    Yields:
        Base: Each nested base object in the hierarchy.
    """
    # Attempt to get the elements attribute, fallback to @elements if necessary
    elements = getattr(base, "elements", getattr(base, "@elements", None))

    if elements is not None:
        for element in elements:
            yield from flatten_base(element)

    yield base


def flatten_base_thorough(base: Base, parent_type: str = None) -> Iterable[Base]:
    """Take a base and flatten it to an iterable of bases.

    Args:
        base: The base object to flatten.
        parent_type: The type of the parent object, if any.

    Yields:
        Base: A flattened base object.
    """
    if isinstance(base, Base):
        base["parent_type"] = parent_type

    elements = getattr(base, "elements", getattr(base, "@elements", None))
    if elements:
        try:
            for element in elements:
                # Recursively yield flattened elements of the child
                yield from flatten_base_thorough(element, base.speckle_type)
        except KeyError:
            pass
    elif hasattr(base, "@Lines"):
        categories = base.get_dynamic_member_names()

        # could be old revit
        try:
            for category in categories:
                print(category)
                if category.startswith("@"):
                    category_object: Base = getattr(base, category)[0]
                    yield from flatten_base_thorough(
                        category_object, category_object.speckle_type
                    )

        except KeyError:
            pass

    else:
        yield base


def extract_base_and_transform(
    base: Base,
    inherited_instance_id: Optional[str] = None,
    transform_list: Optional[List[Transform]] = None,
) -> Tuple[Base, str, Optional[List[Transform]]]:
    """
    Traverses Speckle object hierarchies to yield `Base` objects and their transformations.
    Tailored to Speckle's AEC data structures, it covers the newer hierarchical structures
    with Collections and also  with patterns found in older Revit specific data.

    Parameters:
    - base (Base): The starting point `Base` object for traversal.
    - inherited_instance_id (str, optional): The inherited identifier for `Base` objects without a unique ID.
    - transform_list (List[Transform], optional): Accumulated list of transformations from parent to child objects.

    Yields:
    - tuple: A `Base` object, its identifier, and a list of applicable `Transform` objects or None.

    The id of the `Base` object is either the inherited identifier for a definition from an instance
    or the one defined in the object.
    """
    # Derive the identifier for the current `Base` object, defaulting to an inherited one if needed.
    current_id = getattr(base, "id", inherited_instance_id)
    transform_list = transform_list or []

    if isinstance(base, Instance):
        # Append transformation data and dive into the definition of `Instance` objects.
        if base.transform:
            transform_list.append(base.transform)
        if base.definition:
            yield from extract_base_and_transform(
                base.definition, current_id, transform_list.copy()
            )
    else:
        # Initial yield for the current `Base` object.
        yield base, current_id, transform_list

        # Process 'elements' and '@elements', typical containers for `Base` objects in AEC models.
        elements_attr = getattr(base, "elements", []) or getattr(base, "@elements", [])
        for element in elements_attr:
            if isinstance(element, Base):
                # Recurse into each `Base` object within 'elements' or '@elements'.
                yield from extract_base_and_transform(
                    element, current_id, transform_list.copy()
                )

        # Recursively process '@'-prefixed properties that are Base objects with 'elements'.
        # This is a common pattern in older Speckle data models, such as those used for Revit commits.
        for attr_name in dir(base):
            if attr_name.startswith("@"):
                attr_value = getattr(base, attr_name)
                # If the attribute is a Base object containing 'elements', recurse into it.
                if isinstance(attr_value, Base) and hasattr(attr_value, "elements"):
                    yield from extract_base_and_transform(
                        attr_value, current_id, transform_list.copy()
                    )


def create_mesh_with_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Create normals for a mesh if they do not exist using PyVista.

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        trimesh.Trimesh: The mesh with normals.
    """
    # Convert trimesh to PyVista mesh
    pv_mesh = pv.PolyData(mesh.vertices, mesh.faces.reshape(-1, 4)[:, 1:])

    # Generate normals
    pv_mesh = pv_mesh.compute_normals(
        cell_normals=True, point_normals=True, inplace=True
    )

    # Convert back to trimesh
    mesh.vertices = pv_mesh.points
    mesh.faces = pv_mesh.faces.reshape(-1, 3)
    mesh.vertex_normals = pv_mesh.point_normals
    mesh.face_normals = pv_mesh.cell_normals

    return mesh


def identify_hard_edges(
    mesh: trimesh.Trimesh, angle_threshold: float = 30
) -> np.ndarray:
    """
    Identify hard edges in the mesh based on an angle threshold.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        angle_threshold (float): The angle threshold in degrees for identifying hard edges.

    Returns:
        np.ndarray: Boolean array indicating hard edges.
    """
    face_normals = mesh.face_normals

    hard_edges = np.zeros(len(mesh.edges_unique), dtype=bool)

    for i, edge in enumerate(mesh.edges_unique):
        adjacent_faces = np.where(mesh.face_adjacency_edges == i)[0]
        if len(adjacent_faces) == 2:
            face1, face2 = adjacent_faces
            normal1 = face_normals[face1]
            normal2 = face_normals[face2]
            angle = np.degrees(np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0)))
            if angle > angle_threshold:
                hard_edges[i] = True

    return hard_edges


def create_mesh_with_normals_and_edges(
    mesh: trimesh.Trimesh, angle_threshold: float = 30
) -> trimesh.Trimesh:
    """
    Create normals for a mesh, handling hard and soft edges.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        angle_threshold (float): The angle threshold in degrees for identifying hard edges.

    Returns:
        trimesh.Trimesh: The mesh with normals and hard edges handled.
    """
    hard_edges = identify_hard_edges(mesh, angle_threshold)
    pv_mesh = pv.PolyData(mesh.vertices, mesh.faces.reshape(-1, 4)[:, 1:])

    # Split sharp edges by duplicating vertices
    if any(hard_edges):
        pv_mesh = pv_mesh.extract_feature_edges(
            boundary_edges=False,
            feature_edges=True,
            manifold_edges=False,
            feature_angle=angle_threshold,
        )

    # Generate normals
    pv_mesh = pv_mesh.compute_normals(
        cell_normals=True, point_normals=True, inplace=True
    )

    # Convert back to trimesh
    new_vertices = pv_mesh.points
    new_faces = pv_mesh.faces.reshape(-1, 3)
    vertex_normals = pv_mesh.point_normals
    face_normals = pv_mesh.cell_normals

    # Create new mesh with updated vertices and faces
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    new_mesh.vertex_normals = vertex_normals
    new_mesh.face_normals = face_normals

    return new_mesh
