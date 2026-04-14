import vtk
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET


def parse_pvd(pvd_file):
    """
    Parses a PVD (ParaView Data) file to extract time steps and corresponding VTU file names.

    Parameters:
        pvd_file (str): Path to the .pvd file containing a list of VTU datasets over time.

    Returns:
        tuple:
            - timesteps (list of float): List of time step values.
            - files (list of str): List of VTU file names associated with each time step.
    """
    # Read and parse the XML structure of the PVD file
    tree = ET.parse(pvd_file)
    root = tree.getroot()

    # Initialize lists to store time steps and associated VTU file paths
    timesteps = []
    files = []

    # Iterate over all DataSet entries in the Collection section
    for dataset in root.find("Collection").findall("DataSet"):
        # Extract the time step value
        timesteps.append(float(dataset.attrib["timestep"]))
        # Extract the file name associated with the time step
        files.append(dataset.attrib["file"])

    # Return both lists as output
    return timesteps, files


def compute_roi_mean(domain_vtu, roi_vtu, target_vtu, field):
    """
    Computes the mean value of a given field over the ROI defined in roi_vtu,
    mapped onto the domain_vtu for a single VTU file.

    Parameters:
        domain_vtu (str): Path to the domain VTU file.
        roi_vtu (str): Path to the ROI VTU file.
        target_vtu (str): Path to the VTU file containing the field data.
        field (str): Name of the field to average.

    Returns:
        float: Mean value of the field over the ROI.
    """
    # Load ROI mesh and extract node IDs
    roi_mesh = pv.read(roi_vtu)
    roi_ids = roi_mesh.point_data["bulk_node_ids"]

    # Load domain mesh and extract node IDs
    domain_mesh = pv.read(domain_vtu)
    domain_ids = domain_mesh.point_data["bulk_node_ids"]

    # Create mask for ROI points within the domain
    mask = np.isin(domain_ids, roi_ids)

    # Load target mesh and extract field values at ROI points
    target_mesh = pv.read(target_vtu)
    roi_field = target_mesh.point_data[field][mask]

    # Compute and return mean value
    return np.mean(roi_field)


def mean_over_time(pvd_file, domain_vtu, roi_vtu, field):
    """
    Computes the mean value of a field over the ROI for each timestep defined in a PVD file.

    Parameters:
        pvd_file (str): Path to the PVD file listing VTU files over time.
        domain_vtu (str): Path to the domain VTU file.
        roi_vtu (str): Path to the ROI VTU file.
        field (str): Name of the field to average.

    Returns:
        tuple: Arrays of timesteps and corresponding mean field values.
    """
    output = []
    timesteps, files = parse_pvd(pvd_file)

    for file in files:
        mean_field = compute_roi_mean(domain_vtu, roi_vtu, file, field)
        output.append(mean_field)

    return np.array(timesteps), np.array(output)


def find_nearby_points(mesh, start, end, tolerance=1e-3):
    """
    Finds mesh points that are close to a line segment defined by start and end points.

    Parameters:
        mesh (pyvista.UnstructuredGrid): The mesh containing points to search.
        start (array-like): Starting point of the line segment (3D coordinates).
        end (array-like): Ending point of the line segment (3D coordinates).
        tolerance (float, optional): Maximum allowed distance from the line segment.
                                     Default is 1e-3.

    Returns:
        list of int: Indices of mesh points within the tolerance distance from the line segment.
    """
    # Direction and length of the line segment
    direction = end - start
    length = np.linalg.norm(direction)
    direction_normalized = direction / length

    # Points in the mesh
    points = mesh.points
    extracted_ids = []

    # Find points close to the line segment
    for i, p in enumerate(points):
        v = p - start
        proj = np.dot(v, direction_normalized)
        if 0 <= proj <= length:
            dist = np.linalg.norm(v - proj * direction_normalized)
            if dist < tolerance:
                extracted_ids.append(i)

    return extracted_ids


def calculate_wedge_base_inradius(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """
    Calculates the inradius of the triangular base for VTK_WEDGE elements.
    
    This function assumes that the mesh consists of wedge elements where the 
    first three nodes of each cell define the characteristic triangular face 
    relevant for the CFL condition in advection-dominated problems.

    Parameters:
    -----------
    mesh : pyvista.UnstructuredGrid
        The input mesh containing VTK_WEDGE cells.

    Returns:
    --------
    inradii : np.ndarray
        Array of inradii for each wedge element.
    """
    if mesh.n_cells == 0:
        return np.array([])

    # Extract only the wedge cells
    wedge_mask = mesh.celltypes == vtk.VTK_WEDGE
    # Standard VTK_WEDGE has 6 points. PyVista's .cells is a padded array.
    # We reshape to (n_cells, 7) because each wedge entry is [6, p1, p2, p3, p4, p5, p6]
    cells = mesh.cells.reshape(-1, 7)
    
    # Select only the first three nodes (triangular base) for each wedge
    # Index 0 is the padding (count), 1, 2, 3 are the first triangle nodes
    tri_nodes = cells[wedge_mask, 1:4]
    
    # Retrieve coordinate vectors for the three nodes
    p0 = mesh.points[tri_nodes[:, 0]]
    p1 = mesh.points[tri_nodes[:, 1]]
    p2 = mesh.points[tri_nodes[:, 2]]

    # Compute edge lengths of the triangular base
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    # Semi-perimeter s
    s = (a + b + c) / 2.0

    # Area of the triangular base using cross product (magnitude of vector)
    # Area = 0.5 * |(p1-p0) x (p2-p0)|
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    # Inradius r = Area / s
    # Handling potential division by zero for degenerate elements
    inradii = np.divide(area, s, out=np.zeros_like(area), where=s != 0)

    return inradii

# Example usage:
# mesh = pv.read("hydro_simulation.vtu")
# r_in = calculate_wedge_base_inradius(mesh)
# mesh.cell_data["char_length"] = 2 * r_in

