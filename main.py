"""This module contains the business logic of the function.

Use the automation_context module to wrap your function in an Automate context helper
"""
import os
import tempfile
from typing import Union, List

from specklepy.objects.geometry import Mesh as SpeckleMesh

from speckle_automate import (
    AutomateBase,
    AutomationContext,
    execute_automate_function,
)

import trimesh

from Utilities.helpers import (
    speckle_mesh_to_trimesh,
    combine_transform_matrices,
    extract_base_and_transform,
    create_mesh_with_normals,
)

from pydantic import Field


class FunctionInputs(AutomateBase):
    """These are function author defined values.

    Automate will make sure to supply them matching the types specified here.
    Please use the pydantic model schema to define your inputs:
    https://docs.pydantic.dev/latest/usage/models/
    """

    export_binary: bool = Field(
        default=True,
        title="Export as Binary GLTF (GLB)",
        description="Export the file as a binary GLTF (GLB) instead of plain GLTF.",
    )
    compress_meshes: bool = Field(
        default=False,
        title="Compress Meshes",
        description="Apply mesh compression to reduce file size.",
    )


def automate_function(
    automate_context: AutomationContext,
    function_inputs: FunctionInputs,
) -> None:
    """This is an example Speckle Automate function.

    Args:
        automate_context: A context helper object, that carries relevant information
            about the runtime context of this function.
            It gives access to the Speckle project data, that triggered this run.
            It also has convenience methods attach result data to the Speckle model.
        function_inputs: The input values for this function, as defined in the
            FunctionInputs class.
    """
    # Retrieve the version of the Speckle model that triggered this run.
    version_root_object = automate_context.receive_version()

    # Determine the path where the GLTF file conversion was triggered from, we will use this as the artifact name.
    source_model_path = automate_context.automation_run_data.branch_name

    # Ensure the version data exists.
    if not version_root_object:
        raise Exception("The model version does not exist.")

    # List to store all trimesh objects for the GLTF export.
    mesh_objects = []

    # Extract and transform base objects
    for base, current_id, transform_list in extract_base_and_transform(
        version_root_object
    ):
        if isinstance(base, SpeckleMesh):
            # Convert Speckle mesh to trimesh format.
            base_mesh = speckle_mesh_to_trimesh(base)
            base_mesh = create_mesh_with_normals(base_mesh)

            # Apply transformation if any exist.
            if transform_list:
                transform_matrix = combine_transform_matrices(transform_list)
                base_mesh.apply_transform(transform_matrix)
            mesh_objects.append(base_mesh)
        elif hasattr(base, "displayValue"):
            display_values: Union[List[SpeckleMesh], SpeckleMesh] = base.displayValue

            if not isinstance(display_values, list):
                display_values = [display_values]

            for display_value in display_values:
                if isinstance(display_value, SpeckleMesh):
                    base_mesh = speckle_mesh_to_trimesh(display_value)
                    base_mesh = create_mesh_with_normals(base_mesh)

                    if transform_list:
                        transform_matrix = combine_transform_matrices(transform_list)
                        base_mesh.apply_transform(transform_matrix)
                    mesh_objects.append(base_mesh)

    # Create a scene from all the collected mesh objects.
    scene = trimesh.Scene(mesh_objects)

    # Export the scene to a GLTF file with the specified options.
    export_options = {
        "compress": function_inputs.compress_meshes,  # Apply compression if specified
    }

    # Create a temporary directory to store the output GLTF file.
    with tempfile.TemporaryDirectory() as tmp_dirname:
        # Determine the output file extension based on user input.
        output_extension = ".glb" if function_inputs.export_binary else ".gltf"
        output_path = os.path.join(
            tmp_dirname, f"{source_model_path}{output_extension}"
        )

        # Export the scene to a GLTF file at the specified output path.
        scene.export(output_path, **export_options)

        # Attach the exported GLTF file to the Speckle model.
        automate_context.store_file_result(output_path)

        # Mark the automation run as successful with a success message.
        automate_context.mark_run_success("GLTF file exported successfully.")


# make sure to call the function with the executor
if __name__ == "__main__":
    # NOTE: always pass in the automate function by its reference, do not invoke it!

    # pass in the function reference with the inputs schema to the executor
    execute_automate_function(automate_function, FunctionInputs)

    # if the function has no arguments, the executor can handle it like so
    # execute_automate_function(automate_function_without_inputs)
