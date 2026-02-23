import numpy as np
import g2o


# ============================================
# Vertex: 2D trajectory point
# ============================================
class VertexPointXY(g2o.BaseVertex):
    def __init__(self):
        super().__init__(2)

    def set_to_origin_impl(self):
        self._estimate = np.zeros(2)

    def oplus_impl(self, update):
        self._estimate += update


# ============================================
# Unary edge: ESDF obstacle cost
# residual = d_target - d_esdf(x)
# ============================================
class EdgeESDF(g2o.BaseUnaryEdge):
    def __init__(self, sdf_query_func, d_target, weight=1.0):
        super().__init__(1)
        self.sdf_query = sdf_query_func
        self.d_target = d_target
        self.weight = weight

    def compute_error(self):
        x = self.vertex(0).estimate()
        d, grad = self.sdf_query(x)

        # residual
        self._error[0] = np.sqrt(self.weight) * (self.d_target - d)

        self.grad = grad  # store gradient for jacobian

    def linearize_oplus(self):
        # Jacobian: dr/dx = -grad_d
        self._jacobianOplusXi[0, :] = (
            -np.sqrt(self.weight) * self.grad
        )


# ============================================
# Binary edge: smoothness (Laplacian)
# residual = x_i - x_j
# ============================================
class EdgeSmooth(g2o.BaseBinaryEdge):
    def __init__(self, weight=1.0):
        super().__init__(2)
        self.weight = weight

    def compute_error(self):
        xi = self.vertex(0).estimate()
        xj = self.vertex(1).estimate()
        self._error = np.sqrt(self.weight) * (xi - xj)

    def linearize_oplus(self):
        self._jacobianOplusXi = np.sqrt(self.weight) * np.eye(2)
        self._jacobianOplusXj = -np.sqrt(self.weight) * np.eye(2)


# ============================================
# Example SDF query wrapper
# Must return:
#   distance, gradient (2D vector)
# ============================================
def sdf_query_wrapper(x):
    # Replace with your real sdf_query
    d = np.linalg.norm(x - np.array([1.0, 1.0])) - 0.5
    grad = (x - np.array([1.0, 1.0])) / (
        np.linalg.norm(x - np.array([1.0, 1.0])) + 1e-9
    )
    return d, grad


# ============================================
# Build and solve graph
# ============================================
def optimize_traj(traj_init):

    optimizer = g2o.SparseOptimizer()

    solver = g2o.BlockSolverX(
        g2o.LinearSolverDenseX()
    )
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    vertices = []

    # Add vertices
    for i, pt in enumerate(traj_init):
        v = VertexPointXY()
        v.set_id(i)
        v.set_estimate(pt)
        optimizer.add_vertex(v)
        vertices.append(v)

    # Fix endpoints
    vertices[0].set_fixed(True)
    vertices[-1].set_fixed(True)

    # Add ESDF edges
    for i, v in enumerate(vertices):
        edge = EdgeESDF(
            sdf_query_wrapper,
            d_target=0.2,
            weight=1.0
        )
        edge.set_vertex(0, v)
        edge.set_information(np.eye(1))
        optimizer.add_edge(edge)

    # Add smoothness edges
    for i in range(len(vertices) - 1):
        edge = EdgeSmooth(weight=0.1)
        edge.set_vertex(0, vertices[i])
        edge.set_vertex(1, vertices[i + 1])
        edge.set_information(np.eye(2))
        optimizer.add_edge(edge)

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(20)

    # Extract result
    traj_opt = np.array(
        [v.estimate() for v in vertices]
    )

    return traj_opt


if __name__ == "__main__":

    # initial straight line
    traj = np.linspace([0, 0], [2, 2], 20)

    traj_opt = optimize_traj(traj)

    print("Optimized trajectory:")
    print(traj_opt)
