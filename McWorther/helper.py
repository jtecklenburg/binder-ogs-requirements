import numpy as np
import matplotlib.pyplot as plt
import vtk
from IPython.display import display, HTML, Javascript
import pyvista as pv

class plot_with_error:
    def __init__(self, x_ref, y_ref, style, label_ref, xlabel, ylabel, dylabel):

        self.x_ref = x_ref
        self.y_ref = y_ref
        self.label_ref = label_ref
        self.style = style

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.dylabel = dylabel

        self.data = list()

    def append(self, x, y, style, label):

        self.data.append([x, y, style, label])

    def plot(self, xlim=None):

        #plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
        #plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

        x = np.arange(10)

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

        ax0.plot(self.x_ref, self.y_ref, self.style, label=self.label_ref)

        for data in self.data:
            ax0.plot(data[0], data[1], data[2], label=data[3])
 
        #ax0.set_xlabel(self.xlabel)
        ax0.set_ylabel(self.ylabel)
        ax0.legend()

        ax0.yaxis.tick_left()

        # use default parameter in rcParams, not calling tick_right()
        for data in self.data:
            x, y, style, label = data
            dy = y - np.interp(x, self.x_ref, self.y_ref)
            ax1.plot(x, dy, style, label=label)

            ax1.set_xlabel(self.xlabel)
            ax1.set_ylabel(self.dylabel)
            #ax1.legend()

            ax1.set_xlim(xlim)

        plt.show()


def create_1d_mesh(point_a, point_b, num_points, mesh_type='line'):
    points = np.linspace(point_a, point_b, num_points)
    vtk_points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    if mesh_type == 'line':
        for point in points:
            vtk_points.InsertNextPoint(point)
        for i in range(num_points - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            cells.InsertNextCell(line)
        cell_type = vtk.VTK_LINE
    elif mesh_type == 'quad':
        d = np.linalg.norm(np.array(point_b) - np.array(point_a))
        for i in range(num_points - 1):
            p1, p2 = points[i], points[i + 1]
            p3 = p2 + np.array([0, d/num_points, 0])
            p4 = p1 + np.array([0, d/num_points, 0])
            for p in (p1, p2, p3, p4):
                vtk_points.InsertNextPoint(p)
            quad = vtk.vtkQuad()
            for j in range(4):
                quad.GetPointIds().SetId(j, 4*i + j)
            cells.InsertNextCell(quad)
        cell_type = vtk.VTK_QUAD
    else:
        raise ValueError("Invalid mesh_type. Choose 'line' or 'quad'.")

    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(vtk_points)
    mesh.SetCells(cell_type, cells)

    return pv.UnstructuredGrid(mesh)


def create_boundary_line_meshes(point_a, point_b, num_points):
    d = np.linalg.norm(np.array(point_b) - np.array(point_a))
    offset = np.array([0, d/num_points, 0])
    
    def create_line_mesh(start_point):
        points = vtk.vtkPoints()
        points.InsertNextPoint(start_point)
        points.InsertNextPoint(start_point + offset)
        
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)

        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetLines(cells)
        return pv.UnstructuredGrid(mesh)

    return tuple(map(create_line_mesh, [point_a, point_b]))


def format_numbers(x):
    if x == 0:
        return '0.0'
    if 0.0001 <= abs(x) <= 999999:
        return f'{x:.4f}'.rstrip('0').rstrip('.')
    else:
        return f'{x:.4e}'


# Funktion zum Rendern der Tabelle mit LaTeX-Formeln
def render_latex_table(df, latex_column):
    """
    Rendert eine Pandas DataFrame-Tabelle in einem Jupyter Notebook,
    wobei die LaTeX-Formeln korrekt dargestellt werden.
    
    :param df: Pandas DataFrame
    :param latex_column: Name der Spalte mit LaTeX-Formeln
    """

    # Erstelle HTML für die LaTeX-Spalte
    df[latex_column] = df[latex_column].apply(
        lambda x: f"<div style='text-align: center;'>$$ {x} $$</div>"  # MathJax-Syntax einfügen
    )
    
    # Konvertiere den DataFrame in HTML und aktiviere MathJax für die Darstellung
    html_table = df.to_html(escape=False, index=False, float_format=format_numbers)  # Escape deaktivieren, damit HTML gerendert wird
    
    # MathJax aktivieren und Tabelle anzeigen
    display(HTML("""
        <script type="text/javascript" async
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
        """ + html_table))
    
    # Führe MathJax neu aus, um die Formeln zu rendern
    display(Javascript("""
        if (window.MathJax) {
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    """))