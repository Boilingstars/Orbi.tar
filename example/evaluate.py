"""Entry point. Passes this package's models to the evaluator."""

from example.models import OpsAttentionPolicy
from tools.evaluating import launch

MODELS = [OpsAttentionPolicy]


def main() -> None:
    launch(MODELS)


if __name__ == "__main__":
    main()
