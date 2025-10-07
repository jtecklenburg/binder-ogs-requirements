try:
    import ogs.callbacks as OpenGeoSys
except ModuleNotFoundError:
    import OpenGeoSys

rho = 1000.0      # kg/m³
cp = 4180.0       # J/(kg·K)
Q = 0.03 / 2      # m³/s, halbes Gebiet!
well_length = 30  # m 

class HeatExtraction(OpenGeoSys.SourceTerm):
    def getFlux(self, t, coords, primary_vars):
        temperature = primary_vars[0]
        pressure = primary_vars[1]

        print(temperature)
        print(pressure)

        # Wärmefluss pro Länge (W/m)
        q = -(rho * cp * Q * temperature) / well_length # W/m
        dqdT = -(rho * cp * Q) / well_length
        
        return (q, [dqdT, 0.0])


class HeatInjection(OpenGeoSys.SourceTerm):
    def getFlux(self, t, coords, primary_vars):

        temperature = 1
        # Wärmefluss pro Länge (W/m)
        q = (rho * cp * Q * temperature) / well_length # W/m
        
        return (q, [0.0, 0.0])    


heatextraction = HeatExtraction()
heatinjection = HeatInjection()