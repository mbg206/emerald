import numpy as np
from rlbot import flat
from typing import Optional


def vec3_to_ndarray(vec: flat.Vector3):
    return np.array([vec.x, vec.y, vec.z], dtype=np.float32)

def rotator_to_mtx(rot: flat.Rotator):
    pyr = np.array([rot.pitch, rot.yaw, rot.roll], dtype=np.float32)
    cp, cy, cr = np.cos(pyr)
    sp, sy, sr = np.sin(pyr)

    theta = np.zeros((3,3), dtype=pyr.dtype)

    theta[0, 0] = cp * cy
    theta[1, 0] = cp * sy
    theta[2, 0] = sp

    theta[0, 1] = cy * sp * sr - cr * sy
    theta[1, 1] = sy * sp * sr + cr * cy
    theta[2, 1] = -cp * sr

    theta[0, 2] = -cr * cy * sp - sr * sy
    theta[1, 2] = -cr * sy * sp + sr * cy
    theta[2, 2] = cp * cr

    return theta

INV_VEC = np.array([-1, -1, 1], dtype=np.float32)
INV_MTX = np.array([[-1, -1, -1], [-1, -1, -1], [1, 1, 1]], dtype=np.float32)

MIR_VEC = np.array([-1, 1, 1], dtype=np.float32)
MIR_ANGVEL = np.array([1, -1, -1], dtype=np.float32)
MIR_MTX = np.array([[-1, 1, -1], [1, -1, 1], [1, -1, 1]], dtype=np.float32)


class Physics:
    pos: np.ndarray
    vel: np.ndarray
    angvel: np.ndarray
    rot_mtx: np.ndarray

    def __init__(self, phys: Optional[flat.Physics] = None):
        if (phys is not None):
            self.pos = vec3_to_ndarray(phys.location)
            self.vel = vec3_to_ndarray(phys.velocity)
            self.angvel = vec3_to_ndarray(phys.angular_velocity)
            self.rot_mtx = rotator_to_mtx(phys.rotation)

    def invert(self):
        return self._copy(INV_VEC, INV_VEC, INV_MTX)

    def mirror(self):
        return self._copy(MIR_VEC, MIR_ANGVEL, MIR_MTX)

    def _copy(self, inv_vec: np.ndarray, inv_angvel: np.ndarray, inv_mtx: np.ndarray):
        new_phys = Physics()
        new_phys.pos = self.pos * inv_vec
        new_phys.vel = self.vel * inv_vec
        new_phys.angvel = self.angvel * inv_angvel
        new_phys.rot_mtx = self.rot_mtx * inv_mtx
        return new_phys

    
    @property
    def forward(self) -> np.ndarray:
        return self.rot_mtx[:, 0]

    @property
    def up(self) -> np.ndarray:
        return self.rot_mtx[:, 2]