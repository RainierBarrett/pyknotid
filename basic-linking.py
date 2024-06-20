import numpy as np
import pyknotid
import pyknotid.spacecurves
from pyknotid.spacecurves import Link as link
from pyknotid.invariants import alexander
np.random.seed(42)

# default is 120 from the "detanglement-start.gsd"
# which has 120-mers
def get_data(fpath, mol_size=120):
    raw = np.genfromtxt(fpath, skip_header=2)
    molecule_lines = raw.reshape([-1, 120, 3])
    return molecule_lines

data_file = 'detanglement-start.XYZ'

data = get_data(data_file)

print(data, data.shape)


stride = 1

data = data[:,::stride,:]

print(data.shape)

# this magic number is the simulation box side length
links = link.from_periodic_lines(data, shape=[53.7321]*3)
links.smooth(periodic=False)
a=links.linking_number(include_closures=True)
alex=alexander(links.gauss_code().simplify())

links.plot()

print('LINKING NUMBER: ', a)
print('ALEXANDER: ', alex)
