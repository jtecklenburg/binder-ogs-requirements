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
