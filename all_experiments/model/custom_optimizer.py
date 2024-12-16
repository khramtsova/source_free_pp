
import torch
import math


def make_sgd_optimizer_fn(lr):
    def optimizer_fn(model):
        return torch.optim.SGD(model.parameters(), lr=lr)
    return optimizer_fn


def make_cgn_optimizer_fn(extension):
    def optimizer_fn(model):
        return CGNOptimizer(
            model.parameters(),
            extension,
            lr=0.1,
            damping=1e-2,
            maxiter=50,
            tol=0.1,
            atol=1e-6,
        )

    return optimizer_fn


class CGNOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        parameters,
        bp_extension,
        lr=0.1,
        damping=1e-2,
        maxiter=100,
        tol=1e-1,
        atol=1e-8,
    ):
        super().__init__(
            parameters,
            dict(
                lr=lr,
                damping=damping,
                maxiter=maxiter,
                tol=tol,
                atol=atol,
                savefield=bp_extension.savefield,
            ),
        )
        self.bp_extension = bp_extension

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                print("Parameter p: ", p.shape)
                print("savefield", group["savefield"])
                damped_curvature = self.damped_matvec(
                    p, group["damping"], group["savefield"]
                )

                direction, info = self.cg(
                    damped_curvature,
                    -p.grad.data,
                    maxiter=group["maxiter"],
                    tol=group["tol"],
                    atol=group["atol"],
                )

                print("info: ", info)
                print("direction: ", direction.shape)
                print("p.grad.data: ", p.grad.data.shape)
                p.data.add_(direction, alpha=group["lr"])

    def damped_matvec(self, param, damping, savefield):
        curvprod_fn = getattr(param, savefield)

        def matvec(v):
            v = v.unsqueeze(0)
            result = damping * v + curvprod_fn(v)
            return result.squeeze(0)

        return matvec

    @staticmethod
    def cg(A, b, x0=None, maxiter=None, tol=1e-5, atol=1e-8):
        r"""Solve :math:`Ax = b` for :math:`x` using conjugate gradient.

        The interface is similar to CG provided by :code:`scipy.sparse.linalg.cg`.

        The main iteration loop follows the pseudo code from Wikipedia:
            https://en.wikipedia.org/w/index.php?title=Conjugate_gradient_method&oldid=855450922

        Parameters
        ----------
        A : function
            Function implementing matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system.
        x0 : torch.Tensor
            Initialization estimate.
        atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `atol`
        tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `tol`.
        maxiter: int
            Maximum number of iterations.

        Returns
        -------
        x (torch.Tensor): Approximate solution :math:`x` of the linear system
        info (int): Provides convergence information, if CG converges info
                    corresponds to numiter, otherwise info is set to zero.
        """
        maxiter = b.numel() if maxiter is None else min(maxiter, b.numel())
        x = torch.zeros_like(b) if x0 is None else x0

        # initialize parameters
        r = (b - A(x)).detach()
        p = r.clone()
        rs_old = (r ** 2).sum().item()

        # stopping criterion
        norm_bound = max([tol * torch.norm(b).item(), atol])

        def converged(rs, numiter):
            """Check whether CG stops (convergence or steps exceeded)."""
            norm_converged = norm_bound > math.sqrt(rs)
            info = numiter if norm_converged else 0
            iters_exceeded = numiter > maxiter
            return (norm_converged or iters_exceeded), info

        # iterate
        iterations = 0
        while True:
            Ap = A(p).detach()

            alpha = rs_old / (p * Ap).sum().item()
            x.add_(p, alpha=alpha)
            r.sub_(Ap, alpha=alpha)
            rs_new = (r ** 2).sum().item()
            iterations += 1
            stop, info = converged(rs_new, iterations)
            if stop:
                return x, info

            p.mul_(rs_new / rs_old)
            p.add_(r)
            rs_old = rs_new
