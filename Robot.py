import numpy as np

def DH(a, alpha, d, theta):
    """
    Denavit–Hartenberg transformation matrix.
    """
    cth, sth = np.cos(theta), np.sin(theta)
    cal, sal = np.cos(alpha), np.sin(alpha)
    Rz = np.array([
        [ cth, -sth,  0.0, 0.0],
        [ sth,  cth,  0.0, 0.0],
        [ 0.0,  0.0,  1.0, 0.0],
        [ 0.0,  0.0,  0.0, 1.0]
    ], dtype=np.float64)
    Tz = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0,   d],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    Tx = np.array([
        [1.0, 0.0, 0.0,    a],
        [0.0, 1.0, 0.0,  0.0],
        [0.0, 0.0, 1.0,  0.0],
        [0.0, 0.0, 0.0,  1.0]
    ], dtype=np.float64)
    Rx = np.array([
        [1.0,  0.0,   0.0, 0.0],
        [0.0,  cal,  -sal, 0.0],
        [0.0,  sal,   cal, 0.0],
        [0.0,  0.0,   0.0, 1.0]
    ], dtype=np.float64)
    return Rz @ Tz @ Tx @ Rx

class Robot:
    """
    3-DOF serial robot arm with DH parameters.
    Default DH: a=[0,1,0], alpha=[pi/2,pi/2,0], d=[1,0,1].
    """
    def __init__(self, a=None, alpha=None, d=None, q_ranges=None):
        self.a = np.array([0.0, 1.0, 0.0] if a is None else a, dtype=np.float64)
        self.alpha = np.array([np.pi/2, np.pi/2, 0.0] if alpha is None else alpha, dtype=np.float64)
        self.d = np.array([1.0, 0.0, 1.0] if d is None else d, dtype=np.float64)

        if q_ranges is None:
            self.q_ranges = [
                (-np.pi, np.pi),
                (np.deg2rad(15.0), np.deg2rad(165.0)),
                (np.deg2rad(-85.0), np.deg2rad(85.0))
            ]
        else:
            self.q_ranges = [(float(lo), float(hi)) for (lo, hi) in q_ranges]

    def fk(self, q):
        """
        Forward kinematics: returns H03 transform and joint positions.
        """
        q = np.asarray(q, dtype=np.float64).reshape(3,)
        H01 = DH(self.a[0], self.alpha[0], self.d[0], q[0])
        H12 = DH(self.a[1], self.alpha[1], self.d[1], q[1])
        H23 = DH(self.a[2], self.alpha[2], self.d[2], q[2])
        H02 = H01 @ H12
        H03 = H02 @ H23
        origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p0 = origin
        p1 = H01 @ origin
        p2 = H02 @ origin
        p3 = H03 @ origin
        jx = np.array([p0[0], p1[0], p2[0], p3[0]], dtype=np.float64)
        jy = np.array([p0[1], p1[1], p2[1], p3[1]], dtype=np.float64)
        jz = np.array([p0[2], p1[2], p2[2], p3[2]], dtype=np.float64)
        return H03, (jx, jy, jz)

    def ee_pos(self, q):
        """
        Returns end-effector (x,y,z) for joint angles q.
        """
        H03, _ = self.fk(q)
        return H03[:3, 3].copy()

    def within_limits(self, q):
        """
        Check if q is within joint limits.
        """
        q = np.asarray(q, dtype=float).reshape(3,)
        for qi, (lo, hi) in zip(q, self.q_ranges):
            if not (lo <= qi <= hi):
                return False
        return True

    def clamp(self, q):
        """
        Clamp joints to limits.
        """
        q = np.asarray(q, dtype=float).reshape(3,)
        return np.array([np.clip(qi, lo, hi) for qi, (lo, hi) in zip(q, self.q_ranges)], dtype=np.float64)

    def jacobian_fd(self, q, eps=1e-6):
        """
        Numerical Jacobian of end effector's position with respect to joints via finite differences.
        """
        q = np.asarray(q, dtype=np.float64).reshape(3,)
        base_pos = self.ee_pos(q)
        J = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            dq = np.zeros(3, dtype=np.float64); dq[i] = eps
            pos_d = self.ee_pos(q + dq)
            J[:, i] = (pos_d - base_pos) / eps
        return J
