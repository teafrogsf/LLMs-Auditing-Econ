from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ToyModelConfig:
    """Configuration for initializing a ToyModel.

    Attributes
    - score_mu: Mean score on a specific task, must be in (0, 1)
    - score_sigma: Score spread (treated as sigma), must be > 0
    - reward_param: Scaling factor from score to reward, typically > 0
    - token_mu: Expected tokens required to finish the task, must be > 0
    - token_sigma: Spread of tokens required (treated as sigma), must be > 0
    - token_price: Price per token, must be > 0
    - eta: Cost rate; cost_per_token = eta * token_price, must be >= 0
    """

    score_mu: float
    score_sigma: float
    reward_param: float
    token_mu: float
    token_sigma: float
    token_price: float
    eta: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not (0.0 < self.score_mu < 1.0):
            raise ValueError("score_mu must be in the open interval (0, 1)")
        if not (self.score_sigma > 0.0):
            raise ValueError("score_sigma must be > 0")
        if not (self.reward_param > 0.0):
            raise ValueError("reward_param must be > 0")
        if not (self.token_mu > 0.0):
            raise ValueError("token_mu must be > 0")
        if not (self.token_sigma > 0.0):
            raise ValueError("token_sigma must be > 0")
        if not (self.token_price > 0.0):
            raise ValueError("token_price must be > 0")
        if not (self.eta >= 0.0):
            raise ValueError("eta must be >= 0")


class ToyModel:
    """Toy model defined by a ToyModelConfig.

    Only initialization and basic derived attributes are implemented for now.
    """

    def __init__(self, config: ToyModelConfig) -> None:
        self.config: ToyModelConfig = config

        # Core parameters copied for convenience/readability
        self.mean_score: float = config.score_mu
        self.score_sigma: float = config.score_sigma
        self.reward_scale: float = config.reward_param
        self.token_mean: float = config.token_mu
        self.token_sigma: float = config.token_sigma
        self.token_price: float = config.token_price
        self.eta: float = config.eta

        # Derived
        self.cost_per_token: float = self.eta * self.token_price

    def run_single_task(self, rng=None):
        """Sample one task outcome: score, tokens, reward, cost, price, utility.

        - score ~ Normal(mean_score, score_sigma), clipped to [0, 1]
        - tokens ~ LogNormal(m, s) approximated via Normal(token_mean, token_sigma) clipped to >= 1
          For simplicity, use Normal and clip at 1.
        """
        import random
        import math

        if rng is None:
            rng = random

        score = rng.gauss(self.mean_score, self.score_sigma)
        score = max(0.0, min(1.0, score))

        tokens = rng.gauss(self.token_mean, self.token_sigma)
        if tokens < 1:
            tokens = 1
        tokens = float(tokens)

        reward = self.reward_scale * score
        price = tokens * self.token_price
        cost = tokens * self.cost_per_token
        utility = reward - price

        return {
            'score': float(score),
            'tokens': float(tokens),
            'reward': float(reward),
            'price': float(price),
            'cost': float(cost),
            'utility': float(utility),
        }


@dataclass
class ToyProviderConfig:
    """Configuration for initializing a ToyProvider.

    Attributes
    - num_models: Number of models managed by the provider, must be >= 1
    - model_config: Configuration used to instantiate each ToyModel
    - lie_about_model: Whether to misreport model-related info
    - model_lie_stage: At which stage to misreport model info (1-based index)
    - lie_about_token: Whether to misreport token-related info
    - token_lie_stage: At which stage to misreport token info (1-based index)
    """

    num_models: int
    model_config: ToyModelConfig
    lie_about_model: bool = False
    model_lie_stage: Optional[int] = None
    lie_about_token: bool = False
    token_lie_stage: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.num_models, int) or self.num_models < 1:
            raise ValueError("num_models must be an integer >= 1")
        if not isinstance(self.model_config, ToyModelConfig):
            raise TypeError("model_config must be an instance of ToyModelConfig")

        if self.lie_about_model:
            if self.model_lie_stage is None:
                raise ValueError("model_lie_stage must be provided when lie_about_model is True")
            if not isinstance(self.model_lie_stage, int) or self.model_lie_stage < 1:
                raise ValueError("model_lie_stage must be an integer >= 1")
        else:
            if self.model_lie_stage is not None:
                raise ValueError("model_lie_stage must be None when lie_about_model is False")

        if self.lie_about_token:
            if self.token_lie_stage is None:
                raise ValueError("token_lie_stage must be provided when lie_about_token is True")
            if not isinstance(self.token_lie_stage, int) or self.token_lie_stage < 1:
                raise ValueError("token_lie_stage must be an integer >= 1")
        else:
            if self.token_lie_stage is not None:
                raise ValueError("token_lie_stage must be None when lie_about_token is False")


class ToyProvider:
    """Provider that manages N toy models and misreporting strategies.

    Only initialization is implemented for now.
    """

    def __init__(self, config: ToyProviderConfig) -> None:
        self.config: ToyProviderConfig = config

        # Materialize models
        self.models: List[ToyModel] = [ToyModel(config.model_config) for _ in range(config.num_models)]

        # Strategies
        self.lie_about_model: bool = config.lie_about_model
        self.model_lie_stage: Optional[int] = config.model_lie_stage
        self.lie_about_token: bool = config.lie_about_token
        self.token_lie_stage: Optional[int] = config.token_lie_stage


