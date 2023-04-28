import odl
import numpy as np
import math


def build_gemotry():
    imgSize = 256
    offset = 0
    Volsize = (imgSize - 1) / 2  # ((SIZE(I)+1)/2)
    rec_space = odl.uniform_discr(min_pt=[-Volsize + offset, -Volsize + offset],
                                  max_pt=[Volsize - offset, Volsize - offset], shape=[imgSize, imgSize], dtype='float32')

    # -----fan beam flat detector-----
    AngNum = 320    # 640
    DetNum = 641    # 641
    # det_reso = 1.3712 # mm/pixel size
    SAD = 1075
    AID = math.ceil(SAD+math.sqrt(2)*imgSize) - SAD
    rr = math.sqrt(2)*imgSize/2
    tan1 = rr / SAD
    rDu = math.ceil((AID+SAD)*tan1)
    det_reso = (rDu*2+1-0.5)/DetNum     # mm/pixel size
    rDu = rDu / det_reso

    angOffset = np.pi/2
    angle_partition = odl.uniform_partition(angOffset, angOffset+359*np.pi/180, AngNum)

    detector_partition = odl.uniform_partition(-rDu, rDu, DetNum)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=SAD, det_radius=AID)

    # Ray transform
    ray_trafo = odl.tomo.RayTransform(rec_space, geometry)  #,impl='astra_cuda'
    return ray_trafo
