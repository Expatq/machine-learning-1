import numpy as np
from abc import ABC, abstractmethod
from interfaces import (
    LearningRateSchedule,
    AbstractOptimizer,
    LinearRegressionInterface,
)


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * ((self.s0 / (self.s0 + iteration)) ** self.p)


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR(),
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        current_loss = self.model.compute_loss()
        self.model.loss_history = [current_loss]

        while self.iteration < self.max_iter:
            loss_delta = self._step()

            if np.any(np.isnan(loss_delta)):
                break

            if np.sum(loss_delta**2) < self.tolerance:
                break

            new_loss = self.model.compute_loss()
            self.model.loss_history.append(new_loss)


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # Можно использовать атрибуты класса self.model
        # gradient = 2/n * (X^T X w - X^T y)

        gradient = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)

        w_delta = -lr * gradient
        self.model.w += w_delta

        return w_delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        total_objects = self.model.X_train.shape[0]
        random_indices = np.random.choice(
            total_objects, size=self.batch_size, replace=False
        )

        X_batch = self.model.X_train[random_indices]
        y_batch = self.model.y_train[random_indices]

        gradient = self.model.compute_gradients(X_batch=X_batch, y_batch=y_batch)
        lr = self.lr_schedule.get_lr(self.iteration)

        w_delta = -lr * gradient
        self.model.w += w_delta

        return w_delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size
        self.seen = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros(shape=(num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
            self.seen = np.zeros(num_objects, dtype=bool)

        random_indices = np.random.choice(
            num_objects, size=self.batch_size, replace=False
        )

        # for idx in random_indices:
        #     X_object = X_train[idx : idx + 1]
        #     y_object = y_train[idx : idx + 1]

        #     gradient_idx = self.model.compute_gradients(X_object, y_object)

        #     self.grad_sum -= self.grad_memory[idx]
        #     self.grad_sum += gradient_idx

        #     self.grad_memory[idx] = gradient_idx

        X_batch = X_train[random_indices]
        y_batch = y_train[random_indices]

        new_grads = self.model.compute_per_sample_gradients(X_batch, y_batch)
        old_grads = self.grad_memory[random_indices]

        self.grad_sum += np.sum(new_grads - old_grads, axis=0)
        self.grad_memory[random_indices] = new_grads

        self.seen[random_indices] = True
        num_seen = self.seen.sum()

        average_grad = self.grad_sum / num_seen
        lr = self.lr_schedule.get_lr(self.iteration)

        w_delta = -lr * average_grad
        self.model.w += w_delta

        return w_delta


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)

        self.velocity = self.beta * self.velocity + lr * gradient

        w_delta = -self.velocity
        self.model.w += w_delta

        return w_delta


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if (self.m is None) or (self.v is None):
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients()

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        iter_biased = self.iteration + 1
        m_hat = self.m / (1 - self.beta1**iter_biased)
        v_hat = self.v / (1 - self.beta2**iter_biased)

        lr = self.lr_schedule.get_lr(self.iteration)

        w_diff = -(lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        self.model.w += w_diff

        return w_diff


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        if not hasattr(self.model.loss_function, "analytic_solution"):
            raise NotImplementedError(
                "Loss function does not support analytic solution"
            )

        best_w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )

        self.model.w = best_w
        current_loss = self.model.compute_loss()
        self.model.loss_history = [current_loss]
