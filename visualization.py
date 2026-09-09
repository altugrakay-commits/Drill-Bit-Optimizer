# visualization.py
# Converts CadQuery solids to trimesh, combines, and renders with PyVista.

import os
import tempfile
import trimesh
import pyvista as pv
import cadquery as cq
from typing import Optional, Dict, Any

def component_to_trimesh(solid) -> Optional[trimesh.Trimesh]:
    """Export a CadQuery solid to STL and load as trimesh."""
    if solid is None:
        return None
    try:
        if solid.val() is None:
            return None
    except:
        return None
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        cq.exporters.export(solid, tmp.name)
        tmp_path = tmp.name
    mesh = trimesh.load_mesh(tmp_path)
    return mesh

def components_to_meshes(components: Dict[str, Any]) -> Dict[str, trimesh.Trimesh]:
    """Convert each component to a trimesh object."""
    meshes = {}
    for name, solid in components.items():
        mesh = component_to_trimesh(solid)
        if mesh is not None and mesh.vertices.shape[0] > 0:
            meshes[name] = mesh
    return meshes

def combine_meshes(meshes: Dict[str, trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    """Combine all valid trimesh objects into one."""
    all_meshes = [m for m in meshes.values() if m is not None and m.vertices.shape[0] > 0]
    if not all_meshes:
        return None
    if len(all_meshes) == 1:
        return all_meshes[0]
    return trimesh.util.concatenate(all_meshes)

def visualize_components(components: Dict[str, Any], show: bool = True) -> Optional[pv.Plotter]:
    """Render components in PyVista with distinct colours."""
    meshes = components_to_meshes(components)
    if not meshes:
        print("⚠️ No valid meshes to visualise.")
        return None
    plotter = pv.Plotter()
    colors = {
        'body': 'lightblue',
        'blades': 'lightgreen',
        'cutters': 'gold',
        'nozzles': 'red',            # added for nozzle visibility
    }
    for name, mesh in meshes.items():
        try:
            pv_mesh = pv.wrap(mesh)
            if pv_mesh.n_points == 0:
                continue
            plotter.add_mesh(pv_mesh, color=colors.get(name, 'gray'), label=name, show_edges=True)
        except Exception as e:
            print(f"⚠️ Could not add mesh '{name}': {e}")
    plotter.add_legend()        # type: ignore
    plotter.add_axes()          # type: ignore
    plotter.view_isometric()    # type: ignore
    if show:
        plotter.show()
    return plotter

def export_to_fea(components: Dict[str, Any], output_path: str) -> None:
    """Export combined surface mesh to STL."""
    combined = combine_meshes(components_to_meshes(components))
    if combined is None:
        print("No mesh to export.")
        return
    combined.export(output_path)
    print(f"✅ Surface STL exported to: {output_path}")