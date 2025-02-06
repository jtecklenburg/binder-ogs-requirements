import numpy as np
import matplotlib.pyplot as plt

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