import torch
from torch.nn import Parameter

# this part is from the official github repository of the authors:
# https://github.com/CurvCluster/Congregate

"""
This is built on the class of  Hyperboloid Manifold from 
            [Chami, I., et al., Hyperbolic Graph Convolutional Neural Networks, NuerIPS19].
Note that, we generalize the following manifold to any c,
            i.e., Hyperboloid for negative c's and corresponding Hypersphere of similar time-space construction for positive c's
"""

class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def dist(self, p1, p2, c):
        """Distance between pair of points"""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    # the def defined by mine
    def l_inner(self, x, y, keep_dim):
        # Lorentz inner
        raise NotImplementedError

    def induced_distance(self, x, y, c):
        # metric distance
        raise NotImplementedError

    def lorentzian_distance(self, x, y, c):
        # lorzentzian distance
        raise NotImplementedError

    def exp_map_x(self, p, dp, c, is_res_normalize, is_dp_normalize):
        raise NotImplementedError

    def exp_map_zero(self, dp, c, is_res_normalize, is_dp_normalize):
        raise NotImplementedError

    def log_map_x(self, x, y, c, is_tan_normalize):
        raise NotImplementedError

    def log_map_zero(self, y, c, is_tan_normalize):
        raise NotImplementedError

    def matvec_proj(self, m, x, c):
        raise NotImplementedError

    def matvecbias_proj(self, m, x, b, c):
        raise NotImplementedError

    def matvec_regular(self, m, x, c):
        raise NotImplementedError

    def matvecbias_regular(self, m, x, b, c):
        raise NotImplementedError

    def normalize_tangent_zero(self, p_tan, c):
        raise NotImplementedError

    def lorentz_centroid(self, weight, x, c):
        raise NotImplementedError

    def normalize_input(self, x, c):
        raise NotImplementedError

    def normlize_tangent_bias(self, x, c):
        raise NotImplementedError

    def proj_tan_zero(self, u, c):
        raise NotImplementedError

    def lorentz2poincare(self, x, c):
        raise NotImplementedError

    def poincare2lorentz(self, x, c):
        raise NotImplementedError

    def _lambda_x(self, x, c):
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()


class Hyperboloid(Manifold):
    """
    Note that, we generalize the following manifold to any c,
                i.e., Hyperboloid for negative c's and corresponding Hypersphere of similar time-space construction for positive c's
    Original -  Hyperboloid Manifold class.
            for x in (d+1)-dimension Euclidean space
                    -x0^2 + x1^2 + x2^2 + … + xd = -kappa, x0 > 0, c > 0
            negative curvature - 1 / kappa
    Generalized Manifold
            for x in (d+1)-dimension
                    sign(c)x0^2 + x1^2 + x2^2 + … + xd^2 = 1/c, for all c
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.max_norm = 1000
        self.min_norm = 1e-8
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}

    def l_inner(self, x, y, k, keep_dim=False):
        dim = x.dim()
        end = x.size(-1) - 1
        xy = x * y
        xy = torch.cat((torch.sign(k) * xy.narrow(dim-1, 0, 1), xy.narrow(dim-1, 1, end)), dim=dim-1)
        return torch.sum(xy, dim=dim-1, keepdim=keep_dim)

    def induced_distance(self, x, y, k):
        xy_inner = self.l_inner(x, y, k)
        sqrt_k = torch.abs(k) ** 0.5
        if k < 0:
            return 1 / sqrt_k * arcosh(k * xy_inner + self.eps[x.dtype])
        else:
            return 1 / sqrt_k * torch.arccos(torch.abs(k) * xy_inner + self.eps[x.dtype])

    def normalize(self, p, k): 
        """
        Normalize vector to confirm it is located on the hyperboloid
        :param p: [nodes, features(d + 1)]
        :param c: parameter of curvature
        """
        d = p.size(-1) - 1
        narrowed = p.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        if k < 0:
            first = -1 / k + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
            first = torch.sqrt(first)
            return torch.cat((first, narrowed), dim=1)
        else:
            p = p / torch.norm(input=p, p=2)
            p = p / torch.sqrt(k)
            return p

    def proj(self, p, k): 
        return self.normalize(p, k)

    def normalize_tangent(self, p, p_tan, k):
        """
        Normalize tangent vectors to place the vectors satisfies <p, p_tan>_L=0
        :param p: the tangent spaces at p. size:[nodes, feature]
        :param p_tan: the tangent vector in tangent space at p
        """
        d = p_tan.size(1) - 1
        p_tail = p.narrow(1, 1, d)
        p_tan_tail = p_tan.narrow(1, 1, d)
        ptpt = torch.sum(p_tail * p_tan_tail, dim=1, keepdim=True)
        p_head = torch.sqrt(1/k + torch.sum(torch.pow(p_tail, 2), dim=1, keepdim=True) + self.eps[p.dtype])
        return torch.cat((ptpt / p_head, p_tan_tail), dim=1)

    def normalize_tangent_zero(self, p_tan, k):
        zeros = torch.zeros_like(p_tan)
        zeros[:, 0] = 1 / torch.abs(k) ** 0.5
        return self.normalize_tangent(zeros, p_tan, k)

    def exp_map_x(self, p, dp, k, is_res_normalize=False, is_dp_normalize=False):
        if is_dp_normalize:
            dp = self.normalize_tangent(p, dp, k)
        dp_lnorm = self.l_inner(dp, dp, k, keep_dim=True)
        dp_lnorm = torch.sqrt(torch.clamp(dp_lnorm + self.eps[p.dtype], 1e-6))
        dp_lnorm_cut = torch.clamp(dp_lnorm, max=50)
        sqrt_k = torch.abs(k) ** 0.5
        if k < 0:
            res = (torch.cosh(dp_lnorm_cut * sqrt_k) * p) + (torch.sinh(dp_lnorm_cut * sqrt_k) * dp / dp_lnorm) / sqrt_k
        else:
            res = (torch.cos(dp_lnorm_cut * sqrt_k) * p) + (torch.sin(dp_lnorm_cut * sqrt_k) * dp / dp_lnorm) / sqrt_k
        if is_res_normalize:
            res = self.normalize(res, k)
        return res

    def exp_map_zero(self, dp, k, is_res_normalize=False, is_dp_normalize=False):
        zeros = torch.zeros_like(dp)
        zeros[:, 0] = 1 / torch.abs(k) ** 0.5
        return self.exp_map_x(zeros, dp, k, is_res_normalize, is_dp_normalize)
    
    def expmap0(self, p, k):
        return self.exp_map_zero(p, k)

    def log_map_x(self, x, y, k, is_tan_normalize=False):
        """
        Logarithmic map at x: project hyperboloid vectors to a tangent space at x
        :param x: vector on hyperboloid
        :param y: vector to project a tangent space at x
        :param normalize: whether normalize the y_tangent
        :return: y_tangent
        """
        xy_distance = self.induced_distance(x, y, k)
        # print('xy_distance:\n', xy_distance)
        tmp_vector = y + self.l_inner(x, y, k, keep_dim=True) * torch.abs(k) * x
        tmp_norm = torch.sqrt(self.l_inner(tmp_vector, tmp_vector, k) + self.eps[x.dtype])
        y_tan = xy_distance.unsqueeze(-1) / tmp_norm.unsqueeze(-1) * tmp_vector
        if is_tan_normalize:
            y_tan = self.normalize_tangent(x, y_tan, k)
        return y_tan

    def log_map_zero(self, y, k, is_tan_normalize=False):
        zeros = torch.zeros_like(y)
        zeros[:, 0] = 1 / torch.abs(k) ** 0.5
        return self.log_map_x(zeros, y, k, is_tan_normalize)

    def logmap0(self, p, k):
        return self.log_map_zero(p, k)

    def proj_tan(self, u, p, k):
        """
        project vector u into the tangent vector at p
        :param u: the vector in Euclidean space
        :param p: the vector on a hyperboloid
        """
        return u - self.l_inner(u, p, k, keep_dim=True) / self.l_inner(p, p, k, keep_dim=True) * p

    def proj_tan_zero(self, u, k):
        zeros = torch.zeros_like(u)
        # print(zeros)
        zeros[:, 0] = 1 / torch.abs(k) ** 0.5
        return self.proj_tan(u, zeros, k)

    def proj_tan0(self, u, k):
        return self.proj_tan_zero(u, k)

    def normalize_input(self, x, k):
        # print('=====normalize original input===========')
        num_nodes = x.size(0)
        zeros = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        x_tan = torch.cat((zeros, x), dim=1)
        return self.exp_map_zero(x_tan, k)


def cosh(x:torch.Tensor, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x:torch.Tensor, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x:torch.Tensor, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x:torch.Tensor):
    return Arcosh.apply(x)


def arsinh(x:torch.Tensor):
    return Arsinh.apply(x)


def artanh(x:torch.Tensor):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-10)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
