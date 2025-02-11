import numpy as np
import matplotlib.pyplot as plt
import vtk
from IPython.display import display, HTML, Javascript

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


def mesh1d(point_a, point_b, num_points):
    # Create a vtkPoints object to store the points
    points = vtk.vtkPoints()

    # Generate N points between point_a and point_b using numpy.linspace
    x = np.linspace(point_a[0], point_b[0], num_points)
    y = np.linspace(point_a[1], point_b[1], num_points)
    z = np.linspace(point_a[2], point_b[2], num_points)

    # Add the points to the vtkPoints object
    for i in range(num_points):
        points.InsertNextPoint(x[i], y[i], z[i])

    # Create a vtkCellArray to store the lines
    lines = vtk.vtkCellArray()

    # Create lines between adjacent points
    for i in range(num_points - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    # Create an unstructured grid
    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(points)
    unstructured_grid.SetCells(vtk.VTK_LINE, lines)

    return unstructured_grid


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