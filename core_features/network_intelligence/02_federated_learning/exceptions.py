class FederatedLearningError(Exception):
    """Base exception for the federated learning feature."""


class InvalidFederatedUpdateError(FederatedLearningError):
    """Raised when a client update is malformed or unsafe."""


class GlobalModelNotFoundError(FederatedLearningError):
    """Raised when scoring requires a global model that does not exist."""
